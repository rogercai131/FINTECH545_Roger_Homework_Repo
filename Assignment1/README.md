# Assignment 1

Solutions for FINTECH545 Assignment 1, covering moments/distribution fitting (Q1), OLS vs. MLE
regression under Normal/Student's-t errors (Q2), Pearson vs. Spearman correlation (Q3), the
multivariate Normal and conditioning (Q4), and ACF/PACF-based AR/MA model selection (Q5).

## Contents

- `Code.ipynb` — the notebook with all code, plots, and written answers.
- `Code.pdf` — a static export of the executed notebook.
- `problem1.csv` ... `problem5.csv` — the data sets used by each question.
- `requirements.txt` — pinned package versions used to produce the results.

## How to run

1. **Get the code and data.** From the repo root:

   ```bash
   git clone <this-repo-url>
   cd FINTECH545_Roger_Homework_Repo/Assignment1
   ```

2. **Create and activate a virtual environment** (Python 3.12 was used originally; 3.10+ should work):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook.** Either open it interactively:

   ```bash
   jupyter lab Code.ipynb
   ```

   then choose *Run → Run All Cells*; or reproduce it non-interactively from the command line,
   which re-executes every cell top to bottom and overwrites the saved outputs in place:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace Code.ipynb
   ```

The notebook reads `problem1.csv` through `problem5.csv` via relative paths, so it must be run
with `Assignment1/` as the working directory (as set up above).
