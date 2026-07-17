###############################################################################
# svyglm_analysis.R
# 太阳论文 · R survey包重跑加权Logistic回归
# Purpose: Re-run Table 3 analysis using svyglm() + Taylor linearization
#          to properly account for NHANES complex survey design (PSU/STRATA)
#
# Models (same as manuscript Table 3):
#   Model 1: Unadjusted (exercise timing only)
#   Model 2: Adjusted for age, gender, race/ethnicity
#   Model 3: Fully adjusted (+ BMI + log(MVPA))
#
# Weighting scheme:
#   Pooled 2011-2014, using WTMEC4YR = WTMEC2YR / 2 per NHANES conventions
###############################################################################

library(survey)

# ────────────────────────────────────────────────────────────────────────────
# 1. Load data and merge with raw DEMO weights
# ────────────────────────────────────────────────────────────────────────────
cat("=", rep("─", 68), "=\n", sep="")
cat("  svyglm Analysis - Survey-Weighted Logistic Regression\n")
cat("  Pooled NHANES 2011-2014\n")
cat("=", rep("─", 68), "=\n\n", sep="")

# Read analysis data
base_dir <- "/root/.openclaw/workspace/projects/exercise-timing-paper"
df <- read.csv(file.path(base_dir, "03_analysis", "final_analysis_data.csv"))
cat(sprintf("Loaded analysis dataset: %d rows, %d columns\n", nrow(df), ncol(df)))

# Read raw DEMO files to get WTMEC2YR weights
library(foreign)
demo_g <- read.xport(file.path(base_dir, "02_data", "nhanes_raw", "DEMO_G.XPT"))
demo_h <- read.xport(file.path(base_dir, "02_data", "nhanes_raw", "DEMO_H.XPT"))

# Keep only needed weight columns
demo_wt <- rbind(
  demo_g[, c("SEQN", "WTMEC2YR")],
  demo_h[, c("SEQN", "WTMEC2YR")]
)
cat(sprintf("Loaded weights from DEMO_G (%d rows) and DEMO_H (%d rows)\n",
            nrow(demo_g), nrow(demo_h)))

# Merge weights into analysis data
df_merged <- merge(df, demo_wt, by = "SEQN", all.x = TRUE)
cat(sprintf("After merge: %d rows with WTMEC2YR (%.1f%%)\n",
            sum(!is.na(df_merged$WTMEC2YR)),
            100 * sum(!is.na(df_merged$WTMEC2YR)) / nrow(df_merged)))

# Create 4-year pooled weight (NHANES convention: WTMEC2YR / 2)
df_merged$WTMEC4YR <- df_merged$WTMEC2YR / 2

# ────────────────────────────────────────────────────────────────────────────
# 2. Prepare analysis variables
# ────────────────────────────────────────────────────────────────────────────
cat("\n── Preparing variables ──\n")

# Categorize timing (Morning as reference)
df_merged$timing_cat <- factor(df_merged$dominant_timing,
                                levels = c("Morning", "Noon", "Afternoon",
                                           "Evening", "Mixed"))

# Create indicator for prediabetes/metabolic risk stratification
# (Not needed for main model but keeping for context)

# Age centered (per 10 years)
mean_age <- mean(df_merged$RIDAGEYR, na.rm = TRUE)
df_merged$age_centered <- (df_merged$RIDAGEYR - mean_age) / 10
cat(sprintf("  Mean age = %.1f years, age_centered = (age - %.1f) / 10\n",
            mean_age, mean_age))

# BMI centered (per 5 units)
mean_bmi <- mean(df_merged$BMXBMI, na.rm = TRUE)
df_merged$bmi_centered <- (df_merged$BMXBMI - mean_bmi) / 5
cat(sprintf("  Mean BMI = %.1f, bmi_centered = (BMI - %.1f) / 5\n", mean_bmi, mean_bmi))

# log(MVPA + 1)
df_merged$mvpa_log <- log1p(df_merged$Total_MVPA)

# Race/ethnicity indicators
df_merged$race_black <- ifelse(df_merged$RIDRETH1 == 4, 1, 0)
df_merged$race_hispanic <- ifelse(df_merged$RIDRETH1 %in% c(1, 2), 1, 0)

# Ensure factor variables are factors
df_merged$RIAGENDR <- factor(df_merged$RIAGENDR, levels = c(1, 2),
                              labels = c("Male", "Female"))

# ────────────────────────────────────────────────────────────────────────────
# 3. Define survey design
# ────────────────────────────────────────────────────────────────────────────
cat("\n── Defining survey design ──\n")
cat(sprintf("  PSU variable:    SDMVPSU (%d unique values)\n",
            length(unique(df_merged$SDMVPSU))))
cat(sprintf("  STRATA variable: SDMVSTRA (%d unique values)\n",
            length(unique(df_merged$SDMVSTRA))))
cat(sprintf("  Weight variable: WTMEC4YR (pooled 2011-2014)\n"))

# NHANES design: PSU nested within strata
nhanes_design <- svydesign(
  ids     = ~SDMVPSU,        # Primary sampling unit
  strata  = ~SDMVSTRA,       # Stratification variable
  weights = ~WTMEC4YR,       # 4-year pooled weight
  data    = df_merged,
  nest    = TRUE             # PSU numbers are nested within strata
)

cat(sprintf("  Survey design created: %d observations, %d strata, %d PSUs\n",
            nrow(nhanes_design$variables),
            nlevels(factor(nhanes_design$variables$SDMVSTRA)),
            length(unique(nhanes_design$variables$SDMVPSU))))

# ────────────────────────────────────────────────────────────────────────────
# 4. Run Models
# ────────────────────────────────────────────────────────────────────────────
cat("\n", rep("─", 70), "\n", sep="")
cat("  RUNNING MODELS\n")
cat(rep("─", 70), "\n\n", sep="")

# Helper function to extract OR, 95% CI, and p-value
extract_or_ci <- function(model, coef_name) {
  # Use numeric index to avoid name matching issues
  smry <- summary(model)
  all_names <- rownames(smry$coefficients)
  idx <- which(all_names == coef_name)
  if (length(idx) == 0) {
    return(c(OR = NA, CI_lower = NA, CI_upper = NA, P = NA))
  }
  
  beta  <- smry$coefficients[idx, "Estimate"]
  se    <- smry$coefficients[idx, "Std. Error"]
  pval  <- smry$coefficients[idx, "Pr(>|t|)"]
  
  OR        <- exp(beta)
  ci_lower  <- exp(beta - 1.96 * se)
  ci_upper  <- exp(beta + 1.96 * se)
  
  return(c(OR = OR, CI_lower = ci_lower, CI_upper = ci_upper, P = pval))
}

timing_levels <- c("timing_catNoon", "timing_catAfternoon",
                   "timing_catEvening", "timing_catMixed")

# Build results table
results_list <- list()

# ─── Model 1: Unadjusted ───
cat("──► Model 1: Unadjusted (timing only)\n")

model1 <- svyglm(
  has_mets ~ timing_cat,
  design = nhanes_design,
  family = quasibinomial()   # svyglm uses quasibinomial for survey weights
)

sink(tmp <- tempfile())
sm1 <- summary(model1)
sink()
file.remove(tmp)

cat(sprintf("  N (complete cases) = %d\n", nrow(model1$data)))
cat("  Coefficients:\n")
for (cc in names(coef(model1))) {
  cat(sprintf("    %s: %.4f\n", cc, coef(model1)[cc]))
}

for (tm in timing_levels) {
  res <- extract_or_ci(model1, tm)
  results_list <- append(results_list, list(
    data.frame(Model = "Model 1", Timing = gsub("timing_cat", "", tm),
               OR = res["OR"], CI_lower = res["CI_lower"],
               CI_upper = res["CI_upper"], P = res["P"],
               row.names = NULL)
  ))
  
  cat(sprintf("    %-15s  OR=%.3f  (%.3f–%.3f)  p=%.4f\n",
              gsub("timing_cat", "", tm),
              res["OR"], res["CI_lower"], res["CI_upper"], res["P"]))
}

cat("\n")

# ─── Model 2: Age + Gender + Race ───
cat("──► Model 2: Adjusted for age + gender + race/ethnicity\n")

model2 <- svyglm(
  has_mets ~ timing_cat + age_centered + RIAGENDR + race_black + race_hispanic,
  design = nhanes_design,
  family = quasibinomial()
)

cat(sprintf("  N (complete cases) = %d\n", nrow(model2$data)))

for (tm in timing_levels) {
  res <- extract_or_ci(model2, tm)
  results_list <- append(results_list, list(
    data.frame(Model = "Model 2", Timing = gsub("timing_cat", "", tm),
               OR = res["OR"], CI_lower = res["CI_lower"],
               CI_upper = res["CI_upper"], P = res["P"],
               row.names = NULL)
  ))
  
  cat(sprintf("    %-15s  OR=%.3f  (%.3f–%.3f)  p=%.4f\n",
              gsub("timing_cat", "", tm),
              res["OR"], res["CI_lower"], res["CI_upper"], res["P"]))
}

cat("\n")

# ─── Model 3: Fully adjusted ───
cat("──► Model 3: Fully adjusted (+ BMI + log(MVPA))\n")

model3 <- svyglm(
  has_mets ~ timing_cat + age_centered + RIAGENDR + race_black + race_hispanic +
             bmi_centered + mvpa_log,
  design = nhanes_design,
  family = quasibinomial()
)

cat(sprintf("  N (complete cases) = %d\n", nrow(model3$data)))

for (tm in timing_levels) {
  res <- extract_or_ci(model3, tm)
  results_list <- append(results_list, list(
    data.frame(Model = "Model 3", Timing = gsub("timing_cat", "", tm),
               OR = res["OR"], CI_lower = res["CI_lower"],
               CI_upper = res["CI_upper"], P = res["P"],
               row.names = NULL)
  ))
  
  cat(sprintf("    %-15s  OR=%.3f  (%.3f–%.3f)  p=%.4f\n",
              gsub("timing_cat", "", tm),
              res["OR"], res["CI_lower"], res["CI_upper"], res["P"]))
}

# ────────────────────────────────────────────────────────────────────────────
# 5. Full model summaries
# ────────────────────────────────────────────────────────────────────────────
cat("\n", rep("─", 70), "\n", sep="")
cat("  MODEL SUMMARIES\n")
cat(rep("─", 70), "\n\n", sep="")

cat("═══ Model 1 Summary ═══\n")
print(summary(model1))

cat("\n═══ Model 2 Summary ═══\n")
print(summary(model2))

cat("\n═══ Model 3 Summary ═══\n")
print(summary(model3))

# ────────────────────────────────────────────────────────────────────────────
# 6. Format & save results
# ────────────────────────────────────────────────────────────────────────────
results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL

# Format for Table 3 style
results_df$OR_CI <- sprintf("%.3f (%.3f–%.3f)",
                             results_df$OR,
                             results_df$CI_lower,
                             results_df$CI_upper)
results_df$P_formatted <- sprintf("%.4f", results_df$P)
results_df$P_formatted[results_df$P < 0.0001] <- "<0.0001"

cat("\n", rep("─", 70), "\n", sep="")
cat("  TABLE 3: Survey-Weighted Logistic Regression Results\n")
cat("  NHANES 2011-2014 Pooled, svyglm() + Taylor Linearization\n")
cat(rep("─", 70), "\n\n", sep="")

# Print in nice format
for (mod in c("Model 1", "Model 2", "Model 3")) {
  sub <- results_df[results_df$Model == mod, ]
  cat(sprintf("── %s ──\n", mod))
  cat(sprintf("  %-15s  %-25s  %s\n", "Timing", "OR (95% CI)", "P-value"))
  cat(sprintf("  %s\n", strrep("─", 55)))
  for (i in 1:nrow(sub)) {
    cat(sprintf("  %-15s  %-25s  %s\n",
                sub$Timing[i], sub$OR_CI[i], sub$P_formatted[i]))
  }
  cat("\n")
}

# Save CSV
output_file <- file.path(base_dir, "review", "round1", "survey_weights",
                         "r_svyglm_results.csv")
write.csv(results_df, output_file, row.names = FALSE)
cat(sprintf("✓ Results saved to: %s\n", output_file))

# Also save a summary-format version suitable for Table 3
cat("\n", rep("─", 70), "\n", sep="")
cat("  TABLE 3: Publication-ready summary\n")
cat(rep("─", 70), "\n\n", sep="")

# Pivot-like display
for (mod in c("Model 1", "Model 2", "Model 3")) {
  sub <- results_df[results_df$Model == mod, ]
  cat(sprintf("**%s** (Ref: Morning)\n", mod))
  cat("| Timing | OR (95% CI) | P |\n")
  cat("|--------|-------------|---|\n")
  for (i in 1:nrow(sub)) {
    cat(sprintf("| %s | %s | %s |\n",
                sub$Timing[i], sub$OR_CI[i], sub$P_formatted[i]))
  }
  cat("\n")
}

# ────────────────────────────────────────────────────────────────────────────
# 7. Compare with previously estimated unweighted results
# ────────────────────────────────────────────────────────────────────────────
cat(rep("─", 70), "\n", sep="")
cat("  COMPARISON: svyglm (Taylor) vs previous weighted results\n")
cat(rep("─", 70), "\n\n", sep="")

# Print sample sizes
cat(sprintf("Total analytic sample: %d\n", nrow(df_merged)))
cat(sprintf("Complete cases Model 1: %d\n", nrow(model1$data)))
cat(sprintf("Complete cases Model 2: %d\n", nrow(model2$data)))
cat(sprintf("Complete cases Model 3: %d\n", nrow(model3$data)))

cat("\n── Done ──\n")
