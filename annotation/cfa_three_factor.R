# Three-factor measurement model on the 2,500-item annotation file.
# The manuscript reports CFI, RMSEA, SRMR, loading ranges, and HTMT.
# This script estimates that model from the coded items. It does not
# constrain the estimates to the printed fit statistics.
#
# install.packages("lavaan")

library(lavaan)

ratings <- read.csv("annotation_item_means.csv")
model <- '
  eci =~ eci_verification + eci_evidence + eci_explanation + eci_boundary +
         eci_alternatives + eci_uncertainty + eci_provenance
  ici =~ ici_solution_directness + ici_code_or_procedure + ici_routine_structure
  quality =~ quality_1 + quality_2 + quality_3 + quality_4 + quality_5
'
fit <- cfa(model, data = ratings, std.lv = TRUE)
summary(fit, fit.measures = TRUE, standardized = TRUE)
