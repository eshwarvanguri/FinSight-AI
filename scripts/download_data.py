"""Download / generate sample data for the project.

Usage:
    python scripts/download_data.py --dataset lending_club
    python scripts/download_data.py --dataset sample_10k
    python scripts/download_data.py --dataset synthetic --n 10000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def generate_synthetic_loans(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic loan data resembling LendingClub schema."""
    rng = np.random.default_rng(seed)

    grades = ["A", "B", "C", "D", "E", "F", "G"]
    purposes = [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "small_business", "car", "medical", "other",
    ]
    home_owners = ["RENT", "MORTGAGE", "OWN", "OTHER"]
    emp_lengths = [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
    ]
    verification = ["Verified", "Source Verified", "Not Verified"]
    terms = ["36 months", "60 months"]
    states = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

    df = pd.DataFrame({
        "loan_amnt": rng.integers(1000, 40000, n),
        "term": rng.choice(terms, n, p=[0.7, 0.3]),
        "int_rate": np.round(rng.uniform(5.5, 28.0, n), 2),
        "installment": np.round(rng.uniform(50, 1200, n), 2),
        "grade": rng.choice(grades, n, p=[0.15, 0.25, 0.25, 0.18, 0.10, 0.05, 0.02]),
        "emp_length": rng.choice(emp_lengths, n),
        "home_ownership": rng.choice(home_owners, n, p=[0.4, 0.45, 0.13, 0.02]),
        "annual_inc": np.round(rng.gamma(2, 30000, n), 2),
        "verification_status": rng.choice(verification, n),
        "purpose": rng.choice(purposes, n),
        "addr_state": rng.choice(states, n),
        "dti": np.round(rng.uniform(0, 40, n), 2),
        "delinq_2yrs": rng.poisson(0.3, n),
        "inq_last_6mths": rng.poisson(0.8, n),
        "open_acc": rng.integers(1, 30, n),
        "pub_rec": rng.poisson(0.1, n),
        "revol_util": np.round(rng.uniform(0, 100, n), 2),
        "total_acc": rng.integers(5, 60, n),
        "mort_acc": rng.poisson(1, n),
        "pub_rec_bankruptcies": rng.poisson(0.05, n),
    })
    df["sub_grade"] = df["grade"] + rng.integers(1, 6, n).astype(str)

    # Target: probability of charge-off based on signals
    risk_score = (
        (df["int_rate"] - 12) * 0.05
        + (df["dti"] - 15) * 0.02
        + (df["revol_util"] - 50) * 0.005
        + df["delinq_2yrs"] * 0.3
        + df["pub_rec_bankruptcies"] * 0.5
        + (df["grade"].map({"A": -0.5, "B": -0.2, "C": 0, "D": 0.3, "E": 0.6, "F": 1.0, "G": 1.4}))
        + rng.normal(0, 0.3, n)
    )
    prob_default = 1 / (1 + np.exp(-risk_score))
    df["loan_status"] = np.where(
        rng.random(n) < prob_default, "Charged Off", "Fully Paid"
    )

    return df


def write_sample_10k_text() -> None:
    """Create a few synthetic 10-K-like text files for the RAG demo."""
    out = DATA_DIR / "filings"
    out.mkdir(parents=True, exist_ok=True)

    samples = {
        "ACME_10K_2024.txt": """ACME CORPORATION
ANNUAL REPORT ON FORM 10-K
For the fiscal year ended December 31, 2024

ITEM 1A. RISK FACTORS

Our business is subject to numerous risks. The most significant are:

1. Macroeconomic conditions: Rising interest rates and inflationary pressures
   may reduce consumer demand for our products. In fiscal 2024, we observed
   a 4.2% softening in discretionary spending categories.

2. Supply chain disruptions: We rely on a limited number of suppliers in
   Southeast Asia for critical components. Geopolitical tensions could
   materially affect our ability to source these inputs.

3. Cybersecurity: A successful cyberattack could result in unauthorized
   access to customer data, regulatory penalties under GDPR/CCPA, and
   reputational harm. Our cybersecurity insurance covers up to $50M.

4. Foreign exchange: Approximately 38% of our revenue is generated outside
   the United States. A strengthening US dollar reduces reported revenues.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue for fiscal 2024 was $4.82 billion, an increase of 8.3% over fiscal
2023. Gross margin improved from 41.2% to 43.7% driven by manufacturing
efficiencies. Operating income was $782 million.

Our long-term debt at year-end was $1.4 billion, and our current ratio was
2.1. The Board approved a $500M share buyback authorization in Q4.
""",
        "ZENITH_10K_2024.txt": """ZENITH INDUSTRIES INC
ANNUAL REPORT — FISCAL YEAR 2024

ITEM 1A. RISK FACTORS

Our exposure to credit risk has increased materially in 2024:

- Accounts receivable from our top 5 customers represent 47% of total AR.
  The default of any single major customer could significantly impair
  our working capital.

- Our debt-to-equity ratio rose to 2.4 from 1.8 in 2023 following the
  acquisition of Pinnacle Holdings. Interest coverage declined to 3.1x.

- Regulatory: Pending litigation in the European Union regarding alleged
  anti-competitive behavior could result in fines up to 4% of global revenue.

ITEM 7. FINANCIAL CONDITION

Total revenue: $2.1B (-3.1% YoY)
Net income: $148M
Cash and equivalents: $312M
Total debt: $1.65B

We expect continued margin pressure in the first half of 2025 due to
elevated input costs and competitive pricing in our core markets.
""",
    }
    for name, content in samples.items():
        (out / name).write_text(content)
    print(f"Created {len(samples)} sample filings in {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["lending_club", "sample_10k", "synthetic"])
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.dataset in {"lending_club", "synthetic"}:
        df = generate_synthetic_loans(args.n)
        out = Path(args.output) if args.output else DATA_DIR / "lending_club.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {len(df):,} rows → {out}")
        print(f"Class balance: {df['loan_status'].value_counts(normalize=True).to_dict()}")
    elif args.dataset == "sample_10k":
        write_sample_10k_text()


if __name__ == "__main__":
    main()
