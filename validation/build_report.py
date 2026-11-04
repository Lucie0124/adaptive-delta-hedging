"""Regenerate academic results and a printable report from saved evidence.

PDF generation requires Pandoc and pdflatex. Optional --render uses Poppler
and Pillow to create previews in the ignored tmp/report directory.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "results/validation/tables"


def label(value: str) -> str:
    return {"daily": "Daily", "weekly": "Weekly"}.get(value, value.replace("adaptive_", "Adaptive "))


def number(value: float, digits: int = 4) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list]) -> str:
    """Format a Markdown table without an optional tabulate dependency."""
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"]*len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def numeric_table(frame: pd.DataFrame, headers: list[str], fields: list[str], prefix=None) -> str:
    rows=[]
    for r in frame.itertuples():
        rows.append((prefix(r) if prefix else []) + [number(getattr(r,k)) for k in fields])
    return table(headers,rows)


def results_document() -> str:
    baseline=pd.read_csv(TAB/"baseline_metrics_with_intervals.csv")
    qcheck=pd.read_csv(TAB/"risk_neutral_premium_invariant.csv")
    pricing=pd.read_csv(ROOT/"results/01_model_validation/pricing_validation.csv").iloc[0]
    mesh=pd.read_csv(ROOT/"results/01_model_validation/frictionless_convergence.csv")
    initial=pd.read_csv(ROOT/"results/02_baseline_hedging/summary.csv")
    pairs=pd.read_csv(TAB/"paired_comparisons.csv")
    decisions=pd.read_csv(ROOT/"results/03_cost_risk_optimisation/risk_budget_decisions.csv")
    convergence=pd.read_csv(TAB/"monte_carlo_path_count_convergence.csv")
    stress=pd.read_csv(ROOT/"results/04_volatility_stress_test/stress_summary.csv")
    controlled=pd.read_csv(TAB/"controlled_hedge_volatility_diagnostic.csv")
    interactions=pd.read_csv(TAB/"volatility_ranking_interactions.csv")
    endpoint=pd.read_csv(TAB/"bootstrap_endpoint_monte_carlo_error.csv")
    account=pd.read_csv(TAB/"accounting_checks.csv")
    context={"ACCOUNT_RESIDUAL":f"{account.filter(regex='residual').to_numpy().max():.2e}"}
    context["PRICING_TABLE"]=table(["Pricing diagnostic","Currency per call"],[
        ["Analytical Black-Scholes",number(pricing.analytical_price,6)],
        ["Risk-neutral Monte Carlo",number(pricing.mc_estimate,6)],
        ["Standard error",number(pricing.standard_error,6)],
        ["95% interval",f"[{number(pricing.ci_low,6)}, {number(pricing.ci_high,6)}]"],
        ["Absolute estimation error",number(pricing.absolute_error,6)]])
    policy=lambda r:[label(r.policy_id)]
    context["MESH_TABLE"]=numeric_table(mesh,["Intervals","Mean gross P&L","P&L SD","MAE","RMSE"],
        ["mean_pnl","std_pnl","mae","rmse"],lambda r:[int(r.n_steps)])
    context["QCHECK_TABLE"]=numeric_table(qcheck,["Policy","Expected gross","Observed gross","SE","z error"],
        ["expected_gross_pnl","mean_gross_pnl","se","z_error"],policy)
    context["INITIAL_TABLE"]=numeric_table(initial,["Fee case","Policy","Mean net P&L","SD","RMSE","Mean cost"],
        ["mean_pnl_net","std_pnl_net","rmse_net","mean_total_transaction_cost"],lambda r:[r.fee_case.replace("_"," "),label(r.strategy)])
    context["BASELINE_TABLE"]=numeric_table(baseline,["Policy","Mean net P&L","SD","Net RMSE","Gross RMSE","Mean cost","ES95"],
        ["mean_pnl","sd_pnl","rmse","gross_rmse","mean_cost","es95"],policy)
    keys=["daily","weekly","adaptive_0.01","adaptive_0.05"]
    subset=baseline[baseline.policy_id.isin(keys)]
    context["CI_TABLE"]=table(["Policy","Statistic","Estimate","95% lower","95% upper"],[
        [label(r.policy_id),description,number(getattr(r,k)),number(getattr(r,k+"_low")),number(getattr(r,k+"_high"))]
        for r in subset.itertuples() for k,description in [("mean_pnl","Mean net P&L"),("mean_cost","Mean cost"),("es95","ES95")]])
    context["TRADE_TABLE"]=numeric_table(subset,["Policy","Notional","Interior rebalances","All tickets"],
        ["mean_notional","mean_rebalances","mean_trades"],policy)
    contrast=lambda r:[label(r.first)+" minus "+label(r.second)]
    context["PAIR_TABLE"]=numeric_table(pairs[pairs.metric=="es95"],["First minus second","ES95 difference","95% lower","95% upper"],
        ["difference","low","high"],contrast)
    choice=lambda r:"None" if pd.isna(r.selected_policy) else label(r.selected_policy)
    context["BUDGET_TABLE"]=numeric_table(decisions[decisions.kappa==.002],["Budget","Frozen choice","Evaluation cost","Evaluation ES95","Headroom"],
        ["mean_cost","es","headroom"],lambda r:[number(r.risk_budget,0),choice(r)])
    context["SENSITIVITY_TABLE"]=numeric_table(decisions[decisions.risk_budget==3],["Half-spread","Frozen choice at budget 3","Evaluation cost","Evaluation ES95"],
        ["mean_cost","es"],lambda r:[number(r.kappa),choice(r)])
    rows=[]
    for n,frame in convergence.groupby("n_paths"):
        values=frame.set_index("policy_id");chosen=values.loc["adaptive_0.05"]
        rows.append([f"{n:,}",*[number(values.loc[k,"es95"]) for k in keys],f"[{number(chosen.es_low)}, {number(chosen.es_high)}]"])
    context["PATH_COUNT_TABLE"]=table(["Paths","Daily ES","Weekly ES","Adaptive .01 ES","Adaptive .05 ES",".05 95% interval"],rows)
    context["ENDPOINT_TABLE"]=table(["Half-spread","Reported upper endpoint","Endpoint simulation range"],[
        [number(r.kappa,3),number(r.published_upper_endpoint),f"[{number(r.endpoint_mc_low)}, {number(r.endpoint_mc_high)}]"] for r in endpoint.itertuples()])
    context["STRESS_TABLE"]=numeric_table(stress[stress.policy_id.isin(keys)],["True vol.","Policy","Mean net P&L","SD","Mean cost","ES95"],
        ["mean_pnl_net","std_pnl_net","mean_cost","es"],lambda r:[number(r.sigma_true,2),label(r.policy_id)])
    context["INTERACTION_TABLE"]=table(["First minus second","20% ES gap","25% ES gap","Gap change","Change 95% interval"],[
        [*contrast(r),number(r.es_gap_20),number(r.es_gap_25),number(r.gap_change),f"[{number(r.low)}, {number(r.high)}]"] for r in interactions.itertuples()])
    context["CONTROL_TABLE"]=numeric_table(controlled,["Policy","Hedge vol.","Fair-funded mean","Gross RMSE","Net ES95"],
        ["mean_pnl","gross_rmse","es95"],lambda r:[label(r.policy_id),number(r.sigma_model,2)])
    template=(ROOT/"validation/results_template.md").read_text(encoding="utf-8")
    missing=set(re.findall(r"@@([A-Z0-9_]+)@@",template))-context.keys()
    if missing:raise ValueError(f"Missing result tables: {missing}")
    return re.sub(r"@@([A-Z0-9_]+)@@",lambda m:context[m.group(1)],template)


HEADER=r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,amsmath,amssymb}
\usepackage{booktabs,longtable,array,calc,etoolbox}
\usepackage{graphicx,xcolor,hyperref}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue,pdftitle={Adaptive Delta Hedging under Transaction Costs and Risk Constraints}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\setcounter{tocdepth}{1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\BeforeBeginEnvironment{longtable}{\small}
\AfterEndEnvironment{longtable}{\normalsize}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>0.42\textheight0.42\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\title{Adaptive Delta Hedging under Transaction Costs and Risk Constraints}
\author{Academic simulation study}
\date{}
\begin{document}
\maketitle
\begin{abstract}
Seven delta-hedging policies for a short European call are compared by expected
transaction costs under a terminal Expected Shortfall constraint. Exact GBM,
self-financing accounting and independent calibration and evaluation provide
a reproducible benchmark. Adaptive threshold 0.05 is selected within the
tested candidate set at a 0.2\% half-spread and risk budget 3. Baseline
feasibility does not survive the tested volatility misspecification. Results
are conditional on synthetic model assumptions and do not establish market performance.
\end{abstract}
\tableofcontents
\clearpage
"""


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-only",action="store_true")
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    document=results_document()
    (ROOT/"docs/results_and_discussion.md").write_text(document,encoding="utf-8")
    if args.documents_only:
        print("Generated docs/results_and_discussion.md");return
    if not shutil.which("pandoc") or not shutil.which("pdflatex"):
        raise RuntimeError("PDF generation requires Pandoc and pdflatex on PATH.")
    work=ROOT/"tmp/report";work.mkdir(parents=True,exist_ok=True)
    methodology=(ROOT/"docs/methodology.md").read_text(encoding="utf-8")
    method, references=methodology.split("## Relation to the literature and references",1)
    source=method+"\n\n"+document+"\n\n# References and relation to the literature\n"+references
    source=re.sub(r"(?<!!)\[([^\]]+)\]\((?!https?://)[^)]+\)",r"\1",source)
    converted=subprocess.run(["pandoc","--from=markdown+tex_math_dollars","--to=latex","--top-level-division=section"],
        input=source,cwd=ROOT/"docs",capture_output=True,text=True,check=True)
    tex=ROOT/"docs/research_report.tex"
    def keep_table_together(match):
        block=match.group(0).replace(r"\real{0.1429}",r"\real{0.14285}")
        # Each research table fits on one page. A nonfloating box prevents
        # small ledgers from splitting and avoids longtable/figure interactions.
        block=block.replace(r"\begin{longtable}[]",r"\begin{tabular}")
        block=re.sub(r"\\bottomrule\\noalign\{\}\s*\\endlast(?:foot|head)","",block)
        block=block.replace(r"\endhead","")
        block=block.replace(r"\end{longtable}",r"\bottomrule\end{tabular}")
        return (r"\par\noindent\begin{minipage}{\linewidth}\centering\small"+"\n"
                +block+"\n"+r"\end{minipage}\par\medskip"+"\n")
    body=re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}",keep_table_together,converted.stdout,flags=re.S)
    tex.write_text(HEADER+body+"\n\\end{document}\n",encoding="utf-8")
    for n in (1,2,3,4):
        previous=[(work/(tex.stem+suffix)).read_bytes() if (work/(tex.stem+suffix)).exists() else b"" for suffix in (".aux",".toc")]
        result=subprocess.run(["pdflatex","-interaction=nonstopmode","-halt-on-error",f"-output-directory={work}",tex.name],
            cwd=ROOT/"docs",capture_output=True,text=True)
        (work/f"compile_{n}.txt").write_text(result.stdout+result.stderr,encoding="utf-8")
        if result.returncode:
            print(result.stdout[-4500:]);raise RuntimeError(f"LaTeX compilation failed; see {work}")
        current=[(work/(tex.stem+suffix)).read_bytes() for suffix in (".aux",".toc")]
        if n>=2 and current==previous:break
    pdf=ROOT/"docs/research_report.pdf";shutil.copyfile(work/pdf.name,pdf)
    if args.render:
        for old in [*work.glob("page-*.png"),*work.glob("contact_*.png")]:
            old.unlink()
        commands=[["pdfinfo",str(pdf)],["pdftotext","-layout",str(pdf),str(work/"report.txt")],
                  ["pdftoppm","-r","65","-png",str(pdf),str(work/"page")]]
        for command in commands:
            result=subprocess.run(command,capture_output=True,text=True,check=True)
            if command[0]=="pdfinfo":print(result.stdout)
        from PIL import Image, ImageDraw
        pages=sorted(work.glob("page-*.png"))
        for start in range(0,len(pages),12):
            group=pages[start:start+12]
            sheet=Image.new("RGB",(1100,410*((len(group)+3)//4)),"#dddddd");draw=ImageDraw.Draw(sheet)
            for i,p in enumerate(group):
                im=Image.open(p).convert("RGB");im.thumbnail((265,380))
                x,y=(i%4)*275,(i//4)*410
                sheet.paste(im,(x,y+22));draw.text((x+10,y+5),p.stem,fill="black")
            sheet.save(work/f"contact_{start//12+1}.png")
    print(f"Generated {pdf}")


if __name__=="__main__":main()
