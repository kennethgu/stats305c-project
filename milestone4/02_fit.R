library(mgcv)
library(data.table)

cat("Reading model_data_2020.csv...\n")
df <- fread("model_data_2020.csv")
df$CAND_PTY_AFFILIATION <- factor(df$CAND_PTY_AFFILIATION)
df$CAND_OFFICE_ST       <- factor(df$CAND_OFFICE_ST)

structural_terms <- c(
  "(Intercept)", "population_z", "median_income_z",
  "prop_college_degree_or_higher_18plus_z", "hs_z",
  "CAND_PTY_AFFILIATIONREP", "in_state", "log_dist_z"
)

extract_coefs <- function(model, model_name) {
  if (inherits(model, "gam")) {
    ct <- as.data.frame(summary(model)$p.table)
  } else {
    ct <- as.data.frame(summary(model)$coefficients)
  }
  names(ct) <- c("estimate","std_error","t_value","p_value")
  ct$term  <- rownames(ct)
  ct$model <- model_name
  ct$exp_b_minus_1 <- exp(ct$estimate) - 1
  ct[ct$term %in% structural_terms,
     c("model","term","estimate","std_error","t_value","p_value","exp_b_minus_1")]
}

# --- Model A: OLS with distance (no spatial) ---
cat("Fitting OLS baseline...\n"); t0 <- proc.time()
m_ols <- lm(
  LOG_DONATIONS ~ population_z + median_income_z +
    prop_college_degree_or_higher_18plus_z + hs_z +
    CAND_PTY_AFFILIATION + in_state + log_dist_z,
  data = df
)
cat("  done in", round((proc.time()-t0)["elapsed"],1), "sec\n")

# --- Model B: GP spatial (headline) ---
cat("Fitting GP spatial model (k=200)...\n"); t0 <- proc.time()
m_gp <- bam(
  LOG_DONATIONS ~ population_z + median_income_z +
    prop_college_degree_or_higher_18plus_z + hs_z +
    CAND_PTY_AFFILIATION + in_state + log_dist_z +
    s(lon, lat, bs = "gp", k = 200),
  data = df, discrete = TRUE, nthreads = 4
)
cat("  done in", round((proc.time()-t0)["elapsed"],1), "sec\n")

# --- Model C: GP spatial + destination FE ---
cat("Fitting GP spatial + destination FE...\n"); t0 <- proc.time()
m_gp_fe <- bam(
  LOG_DONATIONS ~ population_z + median_income_z +
    prop_college_degree_or_higher_18plus_z + hs_z +
    CAND_PTY_AFFILIATION + in_state + log_dist_z +
    CAND_OFFICE_ST +
    s(lon, lat, bs = "gp", k = 200),
  data = df, discrete = TRUE, nthreads = 4
)
cat("  done in", round((proc.time()-t0)["elapsed"],1), "sec\n")

# --- Coefficient comparison ---
coef_all <- rbind(
  extract_coefs(m_ols,    "A_OLS_dist"),
  extract_coefs(m_gp,     "B_GP_spatial"),
  extract_coefs(m_gp_fe,  "C_GP_spatial_destFE")
)
write.csv(coef_all, "coef_comparison.csv", row.names = FALSE)

# --- Smooth term table ---
sm_tab <- as.data.frame(summary(m_gp)$s.table)
sm_tab$term <- rownames(sm_tab)
write.csv(sm_tab, "smooth_summary.csv", row.names = FALSE)

# --- Fit statistics ---
fit_stats <- data.frame(
  model        = c("A_OLS_dist", "B_GP_spatial", "C_GP_spatial_destFE"),
  r_squared    = c(summary(m_ols)$r.squared, summary(m_gp)$r.sq,  summary(m_gp_fe)$r.sq),
  dev_explained= c(NA, summary(m_gp)$dev.expl, summary(m_gp_fe)$dev.expl),
  AIC          = c(AIC(m_ols), AIC(m_gp), AIC(m_gp_fe))
)
write.csv(fit_stats, "model_fit_stats.csv", row.names = FALSE)

# --- Spatial field: prediction grid (lon/lat over continental US) ---
grid <- expand.grid(
  lon = seq(-125, -66, length.out = 120),
  lat = seq(  25,  50, length.out = 80)
)
grid$population_z <- 0; grid$median_income_z <- 0
grid$prop_college_degree_or_higher_18plus_z <- 0; grid$hs_z <- 0
grid$CAND_PTY_AFFILIATION <- factor("DEM", levels = levels(df$CAND_PTY_AFFILIATION))
grid$in_state <- 0; grid$log_dist_z <- 0

pred_terms   <- predict(m_gp, newdata = grid, type = "terms")
sp_col       <- grep("s\\(lon", colnames(pred_terms), value = TRUE)[1]
grid$spatial_effect <- pred_terms[, sp_col]
write.csv(grid[, c("lon","lat","spatial_effect")], "spatial_field_grid.csv", row.names = FALSE)

# --- Fitted values for diagnostics ---
# Use predict(newdata=df) instead of fitted() to get full-length output
# (bam with discrete=TRUE drops incomplete cases, so fitted() may be shorter than nrow(df))
df_base <- as.data.frame(df)
df_base$fitted_gp  <- predict(m_gp,  newdata = df_base, type = "response")
df_base$fitted_ols <- predict(m_ols, newdata = df_base)
df_base$resid_gp   <- df_base$LOG_DONATIONS - df_base$fitted_gp
write.csv(df_base[, c("LOG_DONATIONS","fitted_gp","fitted_ols","resid_gp",
                      "ZIP_CODE","CAND_OFFICE_ST","CAND_PTY_AFFILIATION","in_state")],
          "fitted_values_2020.csv", row.names = FALSE)
cat("Wrote fitted_values_2020.csv\n")

# --- 2022 predictions for train/test ---
df22 <- as.data.frame(fread("model_data_2022.csv"))
df22$CAND_PTY_AFFILIATION <- factor(df22$CAND_PTY_AFFILIATION,
                                    levels = levels(df$CAND_PTY_AFFILIATION))
df22$pred_gp  <- predict(m_gp,  newdata = df22, type = "response")
df22$pred_ols <- predict(m_ols, newdata = df22)
write.csv(df22[, c("LOG_DONATIONS","pred_gp","pred_ols",
                   "ZIP_CODE","CAND_OFFICE_ST","CAND_PTY_AFFILIATION","in_state")],
          "preds_2022.csv", row.names = FALSE)
cat("Wrote preds_2022.csv\n")

cat("All outputs written.\n")
