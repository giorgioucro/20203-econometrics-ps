****************************************************
* Problem Set 1 - Tasks 1 to 7
****************************************************

clear all
set more off
version 17

*--------------------------------------------------*
* 0) Load data
*--------------------------------------------------*
* Run this do-file from the repository root so the relative path is portable.
use "assignment1/data.dta", clear

*--------------------------------------------------*
* 1) Describe the data
*--------------------------------------------------*
describe
codebook sector ys w k n
summarize ys w k n, detail

* Sector composition: how many firms in each sector
tabulate sector

* Summary stats by sector
tabstat ys w k n, by(sector) stat(n mean sd min p50 max) columns(statistics)

* Optional correlation table
pwcorr ys w k n, sig star(0.05)

*--------------------------------------------------*
* 2) OLS employment equation: n on w and k
*--------------------------------------------------*
regress n w k
estimates store OLS_base

*--------------------------------------------------*
* 3) Test significance of coefficients on w and k
*    (t-tests are in output; explicit tests below)
*--------------------------------------------------*
test w = 0
test k = 0

*--------------------------------------------------*
* 4) Test homogeneity across sectors using ys and/or
*    sector indicators
*
* Interpretation A: homogeneous intercepts across sectors
* Interpretation B: homogeneous slopes across sectors
* Interpretation X: use ys as additional control
*--------------------------------------------------*

*4A) Sectors indicators (different intercepts + interactions)
regress n c.w##i.sector c.k##i.sector
estimates store FE_slopes_intercepts
testparm i.sector c.w#i.sector c.k#i.sector

* 4B) Sector indicators (different intercepts)
regress n w k i.sector
estimates store FE_intercepts
testparm i.sector

* 4C) Allow slopes to differ across sectors (interactions)
regress n c.w##i.sector c.k##i.sector
estimates store FE_slopes
testparm c.w#i.sector c.k#i.sector

* 4C) Using ys as additional control (we include it in the homogeneity check)
regress n w k ys
estimates store OLS_plus_ys
test ys = 0

*--------------------------------------------------*
* 5) If sector heterogeneity exists, accommodate it
*--------------------------------------------------*

* Example preferred model:
regress n c.w##i.sector c.k##i.sector
estimates store MODEL_heterogeneous

*--------------------------------------------------*
* 6) Test conditional homoskedasticity
*
* Lecture Notes: White test via "estat imtest, white"
* and Breusch-Pagan via "estat hettest"
*--------------------------------------------------*
estat hettest
estat imtest, white

*--------------------------------------------------*
* 7) If heteroskedasticity, address it and rerun with correction
*
* Lecture Notes emphasize robust VCE via vce(robust)
*--------------------------------------------------*
* Baseline OLS (for reference)
regress n c.w##i.sector c.k##i.sector
estimates store MODEL_heterogeneous_OLS

* Correct for heteroskedasticity: robust standard errors
regress n c.w##i.sector c.k##i.sector, vce(robust)
estimates store MODEL_heterogeneous_robust
