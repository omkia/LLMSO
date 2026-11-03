# Valuation and baseline models from the manuscript.
# Install fixest, then point `sample` at the CSV written by draw_stratified_sample.py.
# Clustering is by primary tag. Month fixed effects absorb Post.

sample <- "replication_sample.csv"
d <- read.csv(sample, stringsAsFactors = FALSE)
d$answer_creation_date <- as.POSIXct(d$answer_creation_date, tz = "UTC")
d$post <- as.integer(d$answer_creation_date >= as.POSIXct("2023-01-01", tz = "UTC"))
d$answer_month <- format(d$answer_creation_date, "%Y-%m")
if (!"sampling_weight" %in% names(d)) d$sampling_weight <- 1

library(fixest)

controls <- intersect(c("n_tags", "duplicate_link", "answer_length",
                        "flesch_kincaid", "code_ratio", "formatting_score"), names(d))
rhs <- paste(c(
  "ici + eci + post:ici + post:eci + asi + post:asi",
  "asi:ici + asi:eci + post:asi:ici + post:asi:eci",
  controls
), collapse = " + ")

valuation <- feols(
  as.formula(paste0("net_vote ~ ", rhs, " | user_id + primary_tag + answer_month")),
  data = d,
  weights = ~sampling_weight,
  cluster = ~primary_tag
)
print(summary(valuation))
write.csv(as.data.frame(summary(valuation)$coeftable), "valuation_fixest.csv")
write.csv(as.data.frame(vcov(valuation)), "valuation_fixest_vcov.csv")

baseline <- feols(
  net_vote ~ asi + post:asi | user_id + primary_tag + answer_month,
  data = d,
  weights = ~sampling_weight,
  cluster = ~primary_tag
)
print(summary(baseline))

# Acceptance is a binary response in the diagnostics table.
if ("accepted" %in% names(d)) {
  acceptance <- feglm(
    as.formula(paste0("accepted ~ ", rhs, " | user_id + primary_tag + answer_month")),
    data = d,
    family = binomial(link = "logit"),
    weights = ~sampling_weight,
    cluster = ~primary_tag
  )
  print(summary(acceptance))
}

# Nonnegative counts use Poisson pseudo-maximum likelihood.
if ("comment_count" %in% names(d)) {
  comments <- fepois(
    comment_count ~ asi + post:asi | user_id + primary_tag + answer_month,
    data = d,
    weights = ~sampling_weight,
    cluster = ~primary_tag
  )
  print(summary(comments))
}
