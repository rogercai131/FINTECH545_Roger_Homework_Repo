# Assignment 2

Solutions for FINTECH545 Assignment 2 (Covariance, VaR, and Copulas):

- **Q1:** pairwise vs complete-case correlation and PSD repair.
- **Q2:** VaR after a volatility regime change.
- **Q3:** VaR failing subadditivity.
- **Q4:** Gaussian vs t copula.
- **Q5:** factor-model simulation with correlated residuals.

## Contents

- `Code.ipynb`: the notebook with all code, plots, and the full Predict / Fit / Reconcile write-up for every question.
- `Answers.md` / `Assignment2_Answers.pdf`: the written answers only (the submitted PDF), built from `Answers.md`.
- `figures/`: plots saved by the notebook and embedded in the answers.
- `problem1.csv` ... `problem5.csv`: the data sets used by each question.
- `requirements.txt`: pinned package versions used to produce the results.
- `Library_Gap_Report.md`: notes on which calculations were missing from the shared `Library/` package.
- `pdf_header.tex`: a small LaTeX header used only when building the PDF.
- `assignment.qmd` / `Assignment 2.pdf`: the assignment questions.

**Dependency on `Library/`.** The notebook imports functions from the repo's shared `Library/` folder (one level up). It does this
by adding `../Library` to `sys.path` in its first code cell. The notebook therefore has to be run **from inside this repo**, with
`Assignment2/` as the working directory. Copying the `Assignment2/` folder on its own will not work.

## How to run

1. **Get the code and data.** From the repo root:

   ```bash
   git clone <this-repo-url>
   cd FINTECH545_Roger_Homework_Repo/Assignment2
   ```

2. **Create and activate a virtual environment.** Python 3.12 was used originally; 3.10+ should work.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook.** You can open it interactively:

   ```bash
   jupyter lab Code.ipynb
   ```

   then choose *Run → Run All Cells*. Or you can reproduce it non-interactively from the command line. This re-executes every
   cell top to bottom, overwrites the saved outputs in place, and rewrites the plots in `figures/`:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace Code.ipynb
   ```

   A full run takes about 10 seconds. Run the cells **in order**: later problems reuse helper functions defined earlier,
   e.g. `four_moments` and `historical_var` from Problem 2 are used again in Problems 3–5.

## Reproducibility

- Every number in `Answers.md` / `Assignment2_Answers.pdf` is printed by a cell in `Code.ipynb`. Each problem is a section
  (`# Problem 1` … `# Problem 5`), and each cell is labelled with the question letter it answers.
- The two Monte Carlo simulations use a fixed seed (`545`, 100,000 draws): the copulas in Problem 4(e)–(f) and the factor model
  in Problem 5(c). Re-running gives the same numbers.
- Conventions chosen where the assignment leaves room are commented in the code, mainly:
  - EW volatility is centered at the sample mean.
  - Historical VaR/ES use numpy's default quantile interpolation.
  - Copula parameter counts include the correlations.
  - `ddof=1` is used throughout Problem 5.

## Rebuilding the answers PDF (optional)

`Assignment2_Answers.pdf` is generated from `Answers.md` with [pandoc](https://pandoc.org) and XeLaTeX (needs the DejaVu fonts):

```bash
pandoc Answers.md -o Assignment2_Answers.pdf --pdf-engine=xelatex -H pdf_header.tex \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V mathfont="DejaVu Math TeX Gyre" \
  -V fontsize=10pt -V geometry:margin=0.9in -V colorlinks=true -f markdown-implicit_figures
```

Run the notebook first if you want the embedded plots in `figures/` regenerated.
