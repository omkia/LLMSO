* Answer-level valuation model.
* Net vote: reghdfe OLS, vce(cluster primary_tag).
* Acceptance: logit. Counts: ppmlhdfe.
* Post is absorbed by answer_month. Do not add a Post main effect
* beside saturated month indicators.

version 16
import delimited "replication_sample.csv", clear varnames(1) case(preserve)
gen double answer_time = clock(answer_creation_date, "YMDhms")
format answer_time %tc
gen byte post = answer_time >= clock("2023-01-01 00:00:00", "YMDhms")
gen answer_month = substr(answer_creation_date, 1, 7)
capture confirm variable sampling_weight
if _rc gen sampling_weight = 1

* Equation (valuation), full lower-order expansion, tag-clustered OLS.
reghdfe net_vote c.ici c.eci c.post#c.ici c.post#c.eci ///
    c.asi c.post#c.asi c.asi#c.ici c.asi#c.eci ///
    c.post#c.asi#c.ici c.post#c.asi#c.eci ///
    n_tags duplicate_link ///
    [pw=sampling_weight], ///
    absorb(user_id primary_tag answer_month) vce(cluster primary_tag)
estimates store valuation
estat vce

* Baseline continuous-exposure specification.
reghdfe net_vote c.asi c.post#c.asi [pw=sampling_weight], ///
    absorb(user_id primary_tag answer_month) vce(cluster primary_tag)
estimates store baseline

* Text controls. Drop any variable that is absent from the extract.
capture reghdfe net_vote c.ici c.eci c.post#c.ici c.post#c.eci ///
    c.asi c.post#c.asi c.asi#c.ici c.asi#c.eci ///
    c.post#c.asi#c.ici c.post#c.asi#c.eci ///
    answer_length flesch_kincaid code_ratio formatting_score ///
    [pw=sampling_weight], ///
    absorb(user_id primary_tag answer_month) vce(cluster primary_tag)

capture logit accepted c.ici c.eci c.post#c.ici c.post#c.eci ///
    c.asi c.post#c.asi i.answer_month, vce(cluster primary_tag)

* Contributor-month adaptation, when contributor_month.csv is loaded instead.
* reghdfe eci_share c.post#c.status_pre, absorb(user_id answer_month) vce(cluster user_id)
