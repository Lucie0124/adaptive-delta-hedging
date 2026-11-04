# Numerical validation

The scripts use independent definitions where possible while reusing the
stock paths or fixed policy forms being assessed. They never select on new
evaluation data or alter the numerical engine.

```shell
python validation/run_validation.py
python validation/supplementary_checks.py
python validation/build_report.py --documents-only
```

Run the four main experiments first if their saved outputs are absent. The
first script verifies the discounted-gain ledger identity, independently
recomputes risk metrics and paired intervals, and evaluates fixed policies on
fresh nested samples up to 50,000 paths. The second separates premium funding
from hedge-volatility error, checks a risk-neutral expected-gain identity,
measures bootstrap endpoint simulation noise, and records numerical-domain
limitations. Seeds and sample counts remain explicit in the scripts and saved
metadata. Results are written to `results/validation/`.

Add `--reproduce` to the first command to run pytest and all four experiments
in a temporary directory and compare their main numeric tables with the saved
versions. Any test, accounting, metric or reproduction mismatch aborts the run.
Execution transcripts are local diagnostics ignored by Git; numerical tables,
arrays and compact metadata are retained.

`build_report.py` fills `results_template.md` from numerical CSVs to regenerate
`docs/results_and_discussion.md`. Without `--documents-only`, it also combines
methodology and results into `docs/research_report.tex` and compiles the PDF
using Pandoc and pdflatex. `--render` additionally requires Poppler and Pillow.
Compilation logs and page previews are written under ignored `tmp/report/`.
