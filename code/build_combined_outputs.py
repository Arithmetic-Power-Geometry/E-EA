#!/usr/bin/env python3
"""Generate manuscript-ready combined Experiment-1/Experiment-2 tables and figures."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C=["D","A","O","I","T","R"]

def geom6(v):
    a=np.asarray(v,float)
    return float(np.prod(a)**(1/6))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment1-root",type=Path,required=True,
                    help="Experiment-1 root containing results/tables and results/figures.")
    ap.add_argument("--experiment2-root",type=Path,required=True,
                    help="Experiment-2 result directory containing tables and figures.")
    ap.add_argument("--outdir",type=Path,required=True)
    args=ap.parse_args()
    out=args.outdir; (out/"tables").mkdir(parents=True,exist_ok=True); (out/"figures").mkdir(parents=True,exist_ok=True)

    e1=pd.read_csv(args.experiment1_root/"results/tables/eea_aggregate.csv")
    e1["experiment"]="Experiment 1 (published predictions)"
    e1["source_type"]="historical"
    e1=e1.rename(columns={"model":"model"})
    e2=pd.read_csv(args.experiment2_root/"tables/current_llm_aggregate.csv")
    e2["experiment"]="Experiment 2 (contemporary LLMs)"
    e2["source_type"]="contemporary"

    common=["experiment","source_type","model"]+C+["E_EA_proxy"]
    for extra in ["BBQ_ambig_bias","response_field_JSD"]:
        if extra not in e1: e1[extra]=np.nan
        if extra not in e2: e2[extra]=np.nan
    comb=pd.concat([e1[common+["BBQ_ambig_bias","response_field_JSD"]],
                    e2[common+["BBQ_ambig_bias","response_field_JSD"]]],ignore_index=True)
    comb.to_csv(out/"tables/combined_model_summary.csv",index=False)

    tx=comb.copy()
    for c in C+["E_EA_proxy","BBQ_ambig_bias","response_field_JSD"]:
        tx[c]=tx[c].map(lambda x:"" if pd.isna(x) else f"{x:.3f}")
    (out/"tables/combined_model_summary.tex").write_text(
        tx.to_latex(index=False,escape=True,
                    caption="Combined historical and contemporary E-EA model summaries. Response-field JSD is available only for repeated-generation Experiment 2.",
                    label="tab:combined_models"),
        encoding="utf-8"
    )

    # all-model architecture distance
    X=comb[C].to_numpy(float); names=comb["model"].tolist()
    M=np.zeros((len(names),len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            M[i,j]=np.linalg.norm(X[i]-X[j])/math.sqrt(6)
    dist=pd.DataFrame(M,index=names,columns=names)
    dist.to_csv(out/"tables/combined_architecture_distance.csv")
    d=dist.copy().round(3); d.insert(0,"model",d.index)
    (out/"tables/combined_architecture_distance.tex").write_text(
        d.to_latex(index=False,escape=True,
                   caption="Normalized six-condition distances across historical and contemporary model profiles.",
                   label="tab:combined_distance"),
        encoding="utf-8"
    )

    # profiles
    fig,ax=plt.subplots(figsize=(9,5.2))
    for _,r in comb.iterrows():
        marker="o" if r.source_type=="historical" else "s"
        ax.plot(C,[r[x] for x in C],marker=marker,label=f"{r.model} [{r.source_type}]")
    ax.set_ylim(0,1.02); ax.set_xlabel("E-EA condition"); ax.set_ylabel("Macro-average proxy score")
    ax.legend(fontsize=7,frameon=False,ncol=2)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/"figures"/f"fig_combined_profiles.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)

    # combined bias vs E-EA where available
    fig,ax=plt.subplots(figsize=(7.4,5.0))
    for kind,q in comb.groupby("source_type"):
        z=q.dropna(subset=["BBQ_ambig_bias","E_EA_proxy"])
        ax.scatter(z.BBQ_ambig_bias.abs(),z.E_EA_proxy,label=kind,marker="o" if kind=="historical" else "s")
        for _,r in z.iterrows():
            ax.annotate(r.model,(abs(r.BBQ_ambig_bias),r.E_EA_proxy),fontsize=6,xytext=(3,3),textcoords="offset points")
    ax.set_xlabel("Absolute ambiguous-context BBQ bias")
    ax.set_ylabel("E-EA completeness proxy")
    ax.set_ylim(0,1.02); ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/"figures"/f"fig_combined_bias_vs_eea.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)

    # architecture distance figure
    fig,ax=plt.subplots(figsize=(9,7.4))
    im=ax.imshow(M,vmin=0,vmax=max(.001,float(M.max())))
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=45,ha="right",fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=5)
    fig.colorbar(im,ax=ax,label="Normalized six-condition distance")
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/"figures"/f"fig_combined_architecture_distance.{ext}",dpi=220 if ext=="png" else None,bbox_inches="tight")
    plt.close(fig)

    manifest={
        "tables":[p.name for p in sorted((out/"tables").glob("*"))],
        "figures":[p.name for p in sorted((out/"figures").glob("*"))],
        "n_models":len(comb),
        "historical_models":e1["model"].tolist(),
        "contemporary_models":e2["model"].tolist()
    }
    (out/"publication_output_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Combined manuscript-ready outputs written to {out}")

if __name__=="__main__":
    main()
