"""
Bayesian spatial regression with a Vecchia/NNGP likelihood.

This script is meant to replace the notebook-style dense-GP implementation.
It avoids the PyTensor "loop fusion failed / kernel argument limit" warning by
using padded neighbor arrays plus pytensor.scan, rather than building one giant
symbolic expression with a Python loop over observations.

Model:
    y_i = x_i' beta + w(s_i) + eps_i
    eps_i ~ N(0, tau^2)
    w(.) ~ GP(0, C_theta)

Vecchia approximation:
    p(y | theta) ~= prod_i p(y_i | y_{N_i}, theta)

Default kernel is Matern 3/2.

Examples
--------
Fit only:
    python vecchia_spatial_regression.py fit \
        --input ../data/obs_chosen.csv \
        --sample-n 2000 --m 20 \
        --draws 1000 --tune 1000 --chains 4 --cores 4 \
        --out trace_vecchia.nc

Fit and draw 200 posterior predictive replicates:
    python vecchia_spatial_regression.py fit \
        --input ../data/obs_chosen.csv \
        --sample-n 2000 --m 20 --ppc-draws 200 \
        --out trace_vecchia.nc

Draw posterior predictive later from a saved trace:
    python vecchia_spatial_regression.py ppc \
        --trace trace_vecchia.nc \
        --model-data trace_vecchia_model_data.npz \
        --ppc-draws 500 \
        --out trace_vecchia_with_ppc.nc
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None


DEFAULT_COVARIATES = (
    "prop_college_degree_or_higher_18plus",
    "election_diff",
    "log_contrib_2020",
    "log_income",
    "log_population",
)


@dataclass
class ModelData:
    X: np.ndarray
    y: np.ndarray
    coords_km: np.ndarray
    order: np.ndarray
    neighbor_idx: np.ndarray
    neighbor_mask: np.ndarray
    d_iN: np.ndarray
    d_NN: np.ndarray
    covariates: tuple[str, ...]
    x_means: np.ndarray
    x_sds: np.ndarray


def train_test_split(df: pd.DataFrame, frac: float = 0.8, seed: int = 305):
    train = df.sample(frac=frac, random_state=seed)
    test = df.drop(train.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def lonlat_to_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Project lon/lat to kilometers. Prefer EPSG:5070; fall back to equirectangular."""
    if Transformer is not None:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return np.column_stack([x, y]) / 1000.0

    # Fallback good enough for neighbor construction, not for exact cartography.
    lat0 = np.deg2rad(np.nanmean(lat))
    x = 111.320 * np.cos(lat0) * lon
    y = 110.574 * lat
    return np.column_stack([x, y])


def prepare_arrays(
    df: pd.DataFrame,
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
    sample_n: int | None = None,
    seed: int = 305,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ["log_contrib_2022", "lat", "lon", *covariates]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df.dropna(subset=required).copy().reset_index(drop=True)
    if sample_n is not None and sample_n < len(data):
        data = data.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    x_raw = data[list(covariates)].astype(float).to_numpy()
    x_means = x_raw.mean(axis=0)
    x_sds = x_raw.std(axis=0)
    x_sds[x_sds == 0] = 1.0

    x_scaled = (x_raw - x_means) / x_sds
    X = np.column_stack([np.ones(len(data)), x_scaled]).astype("float64")
    y = data["log_contrib_2022"].astype(float).to_numpy()
    coords_km = lonlat_to_km(
        data["lon"].astype(float).to_numpy(),
        data["lat"].astype(float).to_numpy(),
    ).astype("float64")
    return X, y, coords_km, x_means, x_sds


def spatial_order(coords_km: np.ndarray) -> np.ndarray:
    # Simple deterministic ordering. Max-min ordering can improve approximation quality,
    # but this is much faster and reproducible.
    return np.lexsort((coords_km[:, 1], coords_km[:, 0]))


def make_padded_vecchia_neighbors(coords_km: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return padded nearest-previous-neighbor arrays under a fixed ordering."""
    if m < 1:
        raise ValueError("m must be positive")

    order = spatial_order(coords_km)
    coords = coords_km[order]
    n = coords.shape[0]

    neighbor_idx = np.zeros((n, m), dtype="int64")
    neighbor_mask = np.zeros((n, m), dtype="float64")
    d_iN = np.zeros((n, m), dtype="float64")
    d_NN = np.zeros((n, m, m), dtype="float64")

    # Build trees incrementally enough for clarity. This is preprocessing only.
    for i in range(1, n):
        k = min(m, i)
        tree = cKDTree(coords[:i])
        distances, ids = tree.query(coords[i], k=k)
        ids = np.atleast_1d(ids).astype("int64")
        distances = np.atleast_1d(distances).astype("float64")

        neighbor_idx[i, :k] = ids
        neighbor_mask[i, :k] = 1.0
        d_iN[i, :k] = distances
        d_NN[i, :k, :k] = cdist(coords[ids], coords[ids])

    return order, neighbor_idx, neighbor_mask, d_iN, d_NN


def build_model_data(
    df: pd.DataFrame,
    covariates: tuple[str, ...] = DEFAULT_COVARIATES,
    sample_n: int | None = None,
    m: int = 20,
    seed: int = 305,
) -> ModelData:
    X, y, coords_km, x_means, x_sds = prepare_arrays(
        df=df,
        covariates=covariates,
        sample_n=sample_n,
        seed=seed,
    )
    order, neighbor_idx, neighbor_mask, d_iN, d_NN = make_padded_vecchia_neighbors(coords_km, m=m)

    return ModelData(
        X=X[order],
        y=y[order],
        coords_km=coords_km[order],
        order=order,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        d_iN=d_iN,
        d_NN=d_NN,
        covariates=tuple(covariates),
        x_means=x_means,
        x_sds=x_sds,
    )


def save_model_data(path: str | Path, data: ModelData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=data.X,
        y=data.y,
        coords_km=data.coords_km,
        order=data.order,
        neighbor_idx=data.neighbor_idx,
        neighbor_mask=data.neighbor_mask,
        d_iN=data.d_iN,
        d_NN=data.d_NN,
        covariates=np.array(data.covariates, dtype=object),
        x_means=data.x_means,
        x_sds=data.x_sds,
    )


def load_model_data(path: str | Path) -> ModelData:
    z = np.load(path, allow_pickle=True)
    return ModelData(
        X=z["X"],
        y=z["y"],
        coords_km=z["coords_km"],
        order=z["order"],
        neighbor_idx=z["neighbor_idx"],
        neighbor_mask=z["neighbor_mask"],
        d_iN=z["d_iN"],
        d_NN=z["d_NN"],
        covariates=tuple(z["covariates"].tolist()),
        x_means=z["x_means"],
        x_sds=z["x_sds"],
    )


def kernel_pt(d, sigma2, phi, kernel: str):
    if kernel == "matern32":
        r = np.sqrt(3.0) * phi * d
        return sigma2 * (1.0 + r) * pt.exp(-r)
    if kernel == "expquad":
        return sigma2 * pt.exp(-((phi * d) ** 2))
    raise ValueError(f"unknown kernel: {kernel}")


def kernel_np(d, sigma2: float, phi: float, kernel: str):
    if kernel == "matern32":
        r = np.sqrt(3.0) * phi * d
        return sigma2 * (1.0 + r) * np.exp(-r)
    if kernel == "expquad":
        return sigma2 * np.exp(-((phi * d) ** 2))
    raise ValueError(f"unknown kernel: {kernel}")


def vecchia_loglik_scan(
    y: np.ndarray,
    X: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_mask: np.ndarray,
    d_iN: np.ndarray,
    d_NN: np.ndarray,
    beta,
    sigma2,
    tau2,
    phi,
    kernel: str,
    jitter: float,
):
    """Symbolic Vecchia log likelihood using scan, not a Python expression loop."""
    y_t = pt.as_tensor_variable(y.astype("float64"))
    X_t = pt.as_tensor_variable(X.astype("float64"))
    idx_t = pt.as_tensor_variable(neighbor_idx.astype("int64"))
    mask_t = pt.as_tensor_variable(neighbor_mask.astype("float64"))
    d_iN_t = pt.as_tensor_variable(d_iN.astype("float64"))
    d_NN_t = pt.as_tensor_variable(d_NN.astype("float64"))

    resid = y_t - pt.dot(X_t, beta)
    marginal_var = sigma2 + tau2 + jitter
    m = neighbor_idx.shape[1]
    def one_term(i, idx_i, mask_i, diN_i, dNN_i, resid_all, sigma2_, tau2_, phi_):
        c_iN = kernel_pt(diN_i, sigma2_, phi_, kernel) * mask_i

        mask_outer = mask_i[:, None] * mask_i[None, :]
        C_NN = kernel_pt(dNN_i, sigma2_, phi_, kernel) * mask_outer
        C_NN = C_NN + pt.diag((tau2_ + jitter) * mask_i + (1.0 - mask_i))

        weights = pt.linalg.solve(C_NN, c_iN)
        neigh_resid = resid_all[idx_i] * mask_i
        cond_mean = pt.dot(weights, neigh_resid)
        cond_var = pt.maximum((sigma2_ + tau2_ + jitter) - pt.dot(c_iN, weights), jitter)
        err = resid_all[i] - cond_mean
        return -0.5 * (pt.log(2.0 * np.pi) + pt.log(cond_var) + err**2 / cond_var)

    terms, _ = pytensor.scan(
        fn=one_term,
        sequences=[pt.arange(y.shape[0]), idx_t, mask_t, d_iN_t, d_NN_t],
        non_sequences=[resid, sigma2, tau2, phi],
        strict=True,
    )
    return pt.sum(terms)


def fit_model(
    data: ModelData,
    *,
    kernel: str = "matern32",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    target_accept: float = 0.9,
    beta_sigma: float = 5.0,
    log_sigma2_sigma: float = 5.0,
    nugget_ratio_bounds: tuple[float, float] = (1.0, 100.0),
    inv_phi_bounds: tuple[float, float] = (1.0, 2000.0),
    jitter: float = 1e-6,
    seed: int = 305,
):
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0.0, sigma=beta_sigma, shape=data.X.shape[1])

        log_sigma2 = pm.Normal("log_sigma2", mu=0.0, sigma=log_sigma2_sigma)
        sigma2 = pm.Deterministic("sigma2", pt.exp(log_sigma2))

        nugget_ratio = pm.Uniform(
            "nugget_ratio",
            lower=nugget_ratio_bounds[0],
            upper=nugget_ratio_bounds[1],
        )
        tau2 = pm.Deterministic("tau2", nugget_ratio * sigma2)

        inv_phi = pm.Uniform("inv_phi", lower=inv_phi_bounds[0], upper=inv_phi_bounds[1])
        phi = pm.Deterministic("phi", 1.0 / inv_phi)

        loglik = vecchia_loglik_scan(
            y=data.y,
            X=data.X,
            neighbor_idx=data.neighbor_idx,
            neighbor_mask=data.neighbor_mask,
            d_iN=data.d_iN,
            d_NN=data.d_NN,
            beta=beta,
            sigma2=sigma2,
            tau2=tau2,
            phi=phi,
            kernel=kernel,
            jitter=jitter,
        )
        pm.Potential("vecchia_loglik", loglik)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

    idata.attrs["kernel"] = kernel
    idata.attrs["m_neighbors"] = int(data.neighbor_idx.shape[1])
    idata.attrs["n_observations"] = int(data.y.shape[0])
    idata.attrs["covariates"] = ",".join(data.covariates)
    return idata


def posterior_draw_table(idata: az.InferenceData) -> pd.DataFrame:
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    beta = posterior["beta"].transpose("sample", "beta_dim_0").values
    out = pd.DataFrame(beta, columns=[f"beta_{j}" for j in range(beta.shape[1])])
    for name in ["sigma2", "tau2", "nugget_ratio", "inv_phi", "phi"]:
        out[name] = posterior[name].values
    return out


def simulate_vecchia_yrep(
    data: ModelData,
    beta: np.ndarray,
    sigma2: float,
    tau2: float,
    phi: float,
    *,
    kernel: str,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = data.y.shape[0]
    mu = data.X @ beta
    resid_rep = np.zeros(n)
    y_rep = np.zeros(n)
    marginal_var = sigma2 + tau2 + jitter

    for i in range(n):
        mask = data.neighbor_mask[i].astype(bool)
        if not np.any(mask):
            cond_mean = 0.0
            cond_var = marginal_var
        else:
            idx = data.neighbor_idx[i, mask]
            diN = data.d_iN[i, mask]
            dNN = data.d_NN[i][np.ix_(mask, mask)]

            c_iN = kernel_np(diN, sigma2, phi, kernel)
            C_NN = kernel_np(dNN, sigma2, phi, kernel)
            C_NN = C_NN + (tau2 + jitter) * np.eye(len(idx))

            weights = np.linalg.solve(C_NN, c_iN)
            cond_mean = float(weights @ resid_rep[idx])
            cond_var = float(max(marginal_var - c_iN @ weights, jitter))

        resid_rep[i] = rng.normal(cond_mean, np.sqrt(cond_var))
        y_rep[i] = mu[i] + resid_rep[i]

    return y_rep


def add_posterior_predictive(
    idata: az.InferenceData,
    data: ModelData,
    *,
    ppc_draws: int,
    kernel: str,
    jitter: float = 1e-6,
    seed: int = 306,
) -> az.InferenceData:
    if ppc_draws <= 0:
        return idata

    draws = posterior_draw_table(idata)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(draws), size=min(ppc_draws, len(draws)), replace=False)

    y_rep = np.empty((1, len(chosen), data.y.shape[0]), dtype="float64")
    for j, row_id in enumerate(chosen):
        row = draws.iloc[row_id]
        beta_cols = [c for c in draws.columns if c.startswith("beta_")]
        beta = row[beta_cols].to_numpy(dtype="float64")
        y_rep[0, j, :] = simulate_vecchia_yrep(
            data=data,
            beta=beta,
            sigma2=float(row["sigma2"]),
            tau2=float(row["tau2"]),
            phi=float(row["phi"]),
            kernel=kernel,
            jitter=jitter,
            rng=rng,
        )

    ds = xr.Dataset(
        data_vars={"y_rep": (("chain", "draw", "obs_id"), y_rep)},
        coords={"chain": [0], "draw": np.arange(y_rep.shape[1]), "obs_id": np.arange(data.y.shape[0])},
    )

    if hasattr(idata, "posterior_predictive"):
        idata.posterior_predictive = ds
    else:
        idata.add_groups({"posterior_predictive": ds})
    return idata


def fit_command(args) -> None:
    df = pd.read_csv(args.input)
    train, _ = train_test_split(df, frac=args.train_frac, seed=args.seed)
    covariates = tuple(args.covariates.split(",")) if args.covariates else DEFAULT_COVARIATES

    data = build_model_data(
        train,
        covariates=covariates,
        sample_n=args.sample_n,
        m=args.m,
        seed=args.seed,
    )

    idata = fit_model(
        data,
        kernel=args.kernel,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        beta_sigma=args.beta_sigma,
        nugget_ratio_bounds=(args.nugget_ratio_min, args.nugget_ratio_max),
        inv_phi_bounds=(args.inv_phi_min, args.inv_phi_max),
        jitter=args.jitter,
        seed=args.seed,
    )

    if args.ppc_draws > 0:
        idata = add_posterior_predictive(
            idata,
            data,
            ppc_draws=args.ppc_draws,
            kernel=args.kernel,
            jitter=args.jitter,
            seed=args.seed + 1,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, out)

    model_data_path = Path(args.model_data) if args.model_data else out.with_name(out.stem + "_model_data.npz")
    save_model_data(model_data_path, data)

    print(az.summary(idata, var_names=["beta", "sigma2", "tau2", "inv_phi", "nugget_ratio"]))
    print(f"Saved trace: {out}")
    print(f"Saved model data: {model_data_path}")


def ppc_command(args) -> None:
    idata = az.from_netcdf(args.trace)
    data = load_model_data(args.model_data)
    kernel = args.kernel or idata.attrs.get("kernel", "matern32")

    idata = add_posterior_predictive(
        idata,
        data,
        ppc_draws=args.ppc_draws,
        kernel=kernel,
        jitter=args.jitter,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    az.to_netcdf(idata, out)
    print(f"Saved posterior predictive trace: {out}")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vecchia/NNGP Bayesian spatial regression")
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit the model")
    fit.add_argument("--input", default="../data/obs_chosen.csv")
    fit.add_argument("--out", default="trace_vecchia.nc")
    fit.add_argument("--model-data", default=None, help="Optional .npz path for model data needed by later PPC")
    fit.add_argument("--sample-n", type=int, default=None)
    fit.add_argument("--train-frac", type=float, default=0.8)
    fit.add_argument("--m", type=int, default=20)
    fit.add_argument("--kernel", choices=["matern32", "expquad"], default="matern32")
    fit.add_argument("--draws", type=int, default=1000)
    fit.add_argument("--tune", type=int, default=1000)
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument("--cores", type=int, default=1)
    fit.add_argument("--target-accept", type=float, default=0.9)
    fit.add_argument("--beta-sigma", type=float, default=5.0)
    fit.add_argument("--nugget-ratio-min", type=float, default=1.0)
    fit.add_argument("--nugget-ratio-max", type=float, default=100.0)
    fit.add_argument("--inv-phi-min", type=float, default=1.0)
    fit.add_argument("--inv-phi-max", type=float, default=2000.0)
    fit.add_argument("--jitter", type=float, default=1e-6)
    fit.add_argument("--ppc-draws", type=int, default=0)
    fit.add_argument("--seed", type=int, default=305)
    fit.add_argument("--covariates", default=None, help="Comma-separated covariate list")
    fit.set_defaults(func=fit_command)

    ppc = sub.add_parser("ppc", help="Draw posterior predictive samples from a saved trace")
    ppc.add_argument("--trace", required=True)
    ppc.add_argument("--model-data", required=True)
    ppc.add_argument("--out", default="trace_vecchia_with_ppc.nc")
    ppc.add_argument("--ppc-draws", type=int, default=500)
    ppc.add_argument("--kernel", choices=["matern32", "expquad"], default=None)
    ppc.add_argument("--jitter", type=float, default=1e-6)
    ppc.add_argument("--seed", type=int, default=306)
    ppc.set_defaults(func=ppc_command)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
