library(mgcv); library(data.table)

df22 <- fread("model_data_2022.csv")
df22$CAND_PTY_AFFILIATION <- factor(df22$CAND_PTY_AFFILIATION)
df22$CAND_OFFICE_ST       <- factor(df22$CAND_OFFICE_ST)

cat("Fitting GP spatial model on 2022 data...\n"); t0 <- proc.time()
m_gp22 <- bam(
  LOG_DONATIONS ~ population_z + median_income_z +
    prop_college_degree_or_higher_18plus_z + hs_z +
    CAND_PTY_AFFILIATION + in_state + log_dist_z +
    s(lon, lat, bs = "gp", k = 200),
  data = df22, discrete = TRUE, nthreads = 4
)
cat("  done in", round((proc.time()-t0)["elapsed"],1), "sec\n")

structural_terms <- c(
  "(Intercept)", "population_z", "median_income_z",
  "prop_college_degree_or_higher_18plus_z", "hs_z",
  "CAND_PTY_AFFILIATIONREP", "in_state", "log_dist_z"
)
ct22 <- as.data.frame(summary(m_gp22)$p.table)
names(ct22) <- c("estimate","std_error","t_value","p_value")
ct22$term  <- rownames(ct22)
ct22$model <- "B_GP_spatial_2022"
ct22$exp_b_minus_1 <- exp(ct22$estimate) - 1
ct22 <- ct22[ct22$term %in% structural_terms,
             c("model","term","estimate","std_error","t_value","p_value","exp_b_minus_1")]
write.csv(ct22, "coef_2022.csv", row.names=FALSE)
cat("Wrote coef_2022.csv\n")
