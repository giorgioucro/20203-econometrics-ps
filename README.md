# 20203 Econometrics Problem Sets

This repository contains two group assignments completed for Bocconi University's course **20203 - Econometrics**. It preserves the original assignment instructions, data, code, submitted results, and EViews workfile while adding a portable Python companion for the complete empirical workflow in Assignment 2.

## Assignment 1 - Stata

Assignment 1 studies a simulated cross-section of 3,138 firms in four sectors. The variables are log employment (`n`), log relative wage (`w`), log physical capital (`k`), sectoral output (`ys`), and sector membership (`sector`). The required workflow is to describe the data; estimate an OLS employment equation with `n` on `w` and `k`; test the two slope coefficients; test homogeneity across sectors; accommodate detected sector heterogeneity; test conditional homoskedasticity; and, if needed, correct inference for heteroskedasticity.

The Stata 17 do-file implements descriptive statistics, the baseline OLS model, individual coefficient tests, sector indicators and sector-specific slope interactions, an alternative test using `ys`, Breusch-Pagan and White tests, and heteroskedasticity-robust standard errors. The original dataset and submitted written results are retained beside the instructions.

## Assignment 2 - EViews, with Python replication

Assignment 2 asks for an econometric model of year-on-year real GDP growth and one-step-ahead forecast evaluation over the final ten years of the sample, benchmarked against AR(2) and VAR(2) models. The group selected Italy. The report uses quarterly data for real GDP, HICP inflation, industrial production, the Economic Sentiment Indicator, FTSE MIB returns, and the three-month Euribor. Monthly HICP, ESI, and Euribor observations were aggregated to quarterly frequency in the supplied workbook; growth variables are year-on-year rates.

The submitted analysis uses 1999Q1-2009Q4 for model selection and estimation, followed by forecast evaluation over 2010Q1-2019Q4. Descriptive figures cover 1997Q1-2026Q1. Three static specifications (M1-M3) and four dynamic one-lag ARDL specifications (D1-D4) compare hard, soft, financial, and inflation indicators. D4 - GDP growth on its first lag, current industrial-production growth, and lagged industrial-production growth - is selected as the parsimonious preferred model. The report evaluates heteroskedasticity, fourth-order serial correlation, residual normality, Chow breakpoints at 2008Q4 and 2009Q1, and recursive stability. D4 is then compared with an AR(2) and a three-variable VAR(2) containing GDP growth, HICP inflation, and Euribor.

The original implementation is an **EViews** workfile. No MATLAB source is present. The embedded workfile history contains the model equations, diagnostics, EViews static `fit` forecasts, and error calculations. `companion_code.py` independently recreates this workflow in Python without EViews, MATLAB, Stata, or absolute machine paths.

The principal reported conclusion is that industrial production is the most robust predictor in the candidate regressions, GDP growth is persistent, and the additional soft, financial, and inflation variables do not provide robust incremental explanatory power. For the 2010Q1-2019Q4 one-step-ahead comparison, AR(2) has the lowest RMSFE and MAFE, D4 ranks second, and the more heavily parameterized VAR(2) performs worst.

## Repository structure

```text
20203-econometrics-ps/
|-- .gitignore
|-- README.md
|-- companion_code.py
|-- requirements.txt
|-- assignment1/
|   |-- assignment1_instructions.pdf
|   |-- code.do
|   |-- data.dta
|   `-- results and comments.docx
`-- assignment2/
    |-- assignment2_instructions.pdf
    |-- data_final.xlsx
    |-- PS2_Results_group5.pdf
    |-- PS2_workfile_group5.wf1
    |-- ps2_eviews code_group5.evsettings
    |-- ps2_workfile_group5.evsettings
    `-- output/
        |-- validation_report.txt
        |-- validation_summary.csv
        |-- descriptive_statistics.csv
        |-- correlation_matrix.csv
        |-- model_results.csv
        |-- diagnostic_tests.csv
        |-- residual_statistics.csv
        |-- short_sample_robustness.csv
        |-- ar2_parameters.csv
        |-- var2_parameters.csv
        |-- forecast_accuracy.csv
        |-- forecast_series.csv
        |-- recursive_coefficients.csv
        |-- time_series_hard_indicators.png
        |-- time_series_soft_financial.png
        |-- d4_residual_histogram.png
        |-- d4_recursive_diagnostics.png
        |-- forecast_d4_ar2.png
        `-- forecast_var2.png
```

Key source files retain their original names: `assignment1/assignment1_instructions.pdf`, `assignment2/assignment2_instructions.pdf`, `assignment2/PS2_Results_group5.pdf`, and `assignment2/data_final.xlsx`. The new root-level `companion_code.py` is the portable Python replication.

## Reproducibility

Python 3.10 or newer is recommended. From a newly cloned repository:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and run the companion:

```bash
python -m pip install -r requirements.txt
python companion_code.py
```

The script locates `assignment2/data_final.xlsx` relative to its own file, creates `assignment2/output` if needed, prints labelled results, recreates the report's tables and figures, and writes a row-by-row numerical validation. Compare `assignment2/output/validation_report.txt` and `validation_summary.csv` with `assignment2/PS2_Results_group5.pdf`.

### Replication conventions and limitations

- The report's descriptive table reflects a specific Excel/EViews convention: formula references to blank 2026Q1 GDP, IP, and FTSE MIB cells were cached as zero, while pre-1999 FTSE MIB zeros were reset to missing. The companion reproduces this convention only for slides 5-8 and documents it in code; estimation uses the unmodified `db` data.
- The regressions use growth rates in decimal units. Descriptive statistics and forecast errors are reported in percentage points, matching the submitted slides.
- EViews estimates the VAR(2) on 44 dependent observations from 1999Q1-2009Q4 by obtaining its two lags from 1998Q3-1998Q4. The companion implements the equivalent equation-by-equation OLS system rather than accepting the different sample behavior of a direct `statsmodels.VAR` call.
- All substantive model, residual, breakpoint, and forecast results reproduce the report to its displayed precision except two documented slide transcription errors: D4's current-IP coefficient is 0.3117 rather than 0.3120, and the White-test F p-value is 0.9563 rather than 0.9663. Four information-criterion cells also differ by less than 0.0001 because the presentation table appears to truncate or manually transcribe the final displayed digit. None of these differences changes the conclusions.
- Recursive plots are econometrically equivalent recreations. Plot styling and the interpolation of critical-value lines may not be pixel-identical to EViews.

