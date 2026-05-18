"""
Dataset Inspector for GreenML Benchmark
Run this on Google Colab FIRST before anything else.
It reads all dataset files and saves a full inspection report to your Drive.

Instructions:
  1. Run this entire script in a Colab cell (or upload as .py and run with %run inspect_datasets.py)
  2. It will mount your Drive automatically
  3. Find the report at: MyResearchWork/phdconf/results/dataset_inspection_report.txt
  4. Download and share that file — we use it to rewrite the main notebook correctly.
"""

# ── Step 1: Mount Drive ────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# ── Step 2: Imports ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import os
from pathlib import Path
from io import StringIO

# ── Step 3: Paths ──────────────────────────────────────────────────────────────
BASE       = Path('/content/drive/MyDrive/MyResearchWork/phdconf')
DATA_DIR   = BASE / 'datasets'
RESULTS    = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

REPORT_PATH = RESULTS / 'dataset_inspection_report.txt'

# Files to inspect
FILES = {
    'IEEE_train_transaction' : DATA_DIR / 'IEEE-CIS'  / 'train_transaction.csv',
    'IEEE_train_identity'    : DATA_DIR / 'IEEE-CIS'  / 'train_identity.csv',
    'IEEE_test_transaction'  : DATA_DIR / 'IEEE-CIS'  / 'test_transaction.csv',
    'IEEE_test_identity'     : DATA_DIR / 'IEEE-CIS'  / 'test_identity.csv',
    'HM_articles'            : DATA_DIR / 'HandM'     / 'articles_hm.csv',
}

# ── Step 4: Inspector ──────────────────────────────────────────────────────────
def inspect(name: str, path: Path, nrows_full: int = 5000) -> str:
    out = StringIO()

    def w(line=''):
        out.write(line + '\n')

    w('=' * 80)
    w(f'DATASET: {name}')
    w(f'PATH   : {path}')
    w('=' * 80)

    if not path.exists():
        w(f'  ❌  FILE NOT FOUND: {path}')
        w()
        return out.getvalue()

    # File size
    size_mb = path.stat().st_size / 1e6
    w(f'File size : {size_mb:.1f} MB')

    # Load a sample first to get column info cheaply
    try:
        df_sample = pd.read_csv(path, nrows=nrows_full, low_memory=False)
    except Exception as e:
        w(f'  ❌  Could not read file: {e}')
        return out.getvalue()

    # Count total rows without loading full file
    try:
        total_rows = sum(1 for _ in open(path)) - 1  # subtract header
    except Exception:
        total_rows = '(could not count)'

    w(f'Total rows (approx): {total_rows:,}')
    w(f'Total columns       : {len(df_sample.columns)}')
    w()

    # ── Column overview ────────────────────────────────────────────────────────
    w('── COLUMN OVERVIEW ──────────────────────────────────────────────────────')
    w(f'  {"#":<5} {"Column Name":<35} {"Dtype":<12} {"Nulls%":<10} {"Unique":<10} Sample Values')
    w('  ' + '-' * 100)

    for i, col in enumerate(df_sample.columns):
        series = df_sample[col]
        null_pct = series.isna().mean() * 100
        n_unique = series.nunique()
        # Sample 3 non-null values
        sample_vals = series.dropna().head(3).tolist()
        sample_str  = str(sample_vals)[:60]
        w(f'  {i:<5} {col:<35} {str(series.dtype):<12} {null_pct:<10.1f} {n_unique:<10} {sample_str}')

    w()

    # ── Numeric columns summary ────────────────────────────────────────────────
    num_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        w('── NUMERIC COLUMNS SUMMARY ──────────────────────────────────────────────')
        w(f'  Count: {len(num_cols)}')
        w(f'  Names: {num_cols[:30]}{"..." if len(num_cols) > 30 else ""}')
        w()
        desc = df_sample[num_cols[:20]].describe().round(4)  # first 20 to keep report readable
        w(desc.to_string())
        w()

    # ── Categorical / object columns ───────────────────────────────────────────
    cat_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        w('── CATEGORICAL COLUMNS ──────────────────────────────────────────────────')
        w(f'  Count: {len(cat_cols)}')
        for col in cat_cols[:20]:
            vc = df_sample[col].value_counts().head(5)
            w(f'  {col}: {dict(vc)}')
        if len(cat_cols) > 20:
            w(f'  ... and {len(cat_cols) - 20} more categorical columns')
        w()

    # ── Target / label column detection ───────────────────────────────────────
    w('── LIKELY TARGET COLUMNS ────────────────────────────────────────────────')
    possible_targets = [c for c in df_sample.columns
                        if any(kw in c.lower() for kw in
                               ['fraud', 'label', 'target', 'class', 'purchase',
                                'click', 'buy', 'isFraud', 'y', 'sold'])]
    if possible_targets:
        for col in possible_targets:
            vc = df_sample[col].value_counts()
            w(f'  {col}: {dict(vc)} (dtype={df_sample[col].dtype})')
    else:
        w('  None auto-detected. Check column list above manually.')
    w()

    # ── Missing value summary ──────────────────────────────────────────────────
    missing = df_sample.isna().mean() * 100
    high_missing = missing[missing > 20].sort_values(ascending=False)
    w('── HIGH-MISSING COLUMNS (>20% null) ─────────────────────────────────────')
    if len(high_missing) > 0:
        w(f'  Count: {len(high_missing)}')
        for col, pct in high_missing.head(20).items():
            w(f'  {col}: {pct:.1f}%')
        if len(high_missing) > 20:
            w(f'  ... and {len(high_missing) - 20} more')
    else:
        w('  None — all columns <20% missing. Clean dataset!')
    w()

    # ── Memory usage ───────────────────────────────────────────────────────────
    mem_mb = df_sample.memory_usage(deep=True).sum() / 1e6
    w('── MEMORY (sample only) ─────────────────────────────────────────────────')
    w(f'  {nrows_full:,} rows use {mem_mb:.1f} MB → full dataset estimate: '
      f'{mem_mb * (total_rows / nrows_full if isinstance(total_rows, int) else 1):.0f} MB')
    w()

    return out.getvalue()


# ── Step 5: Run inspection on all files ───────────────────────────────────────
print('Starting dataset inspection...\n')
full_report = []
full_report.append('GREEN ML BENCHMARK — DATASET INSPECTION REPORT\n')
full_report.append(f'Generated on Colab\n')
full_report.append('=' * 80 + '\n\n')

for name, path in FILES.items():
    print(f'  Inspecting {name}...')
    report_section = inspect(name, path)
    full_report.append(report_section)
    print(f'  Done.\n')

# ── Step 6: Write report ───────────────────────────────────────────────────────
final_report = '\n'.join(full_report)
with open(REPORT_PATH, 'w') as f:
    f.write(final_report)

print(f'\n✅ Report saved to:\n   {REPORT_PATH}')
print('\n👉 Download it from Drive and share with Claude to rewrite the notebook.')
