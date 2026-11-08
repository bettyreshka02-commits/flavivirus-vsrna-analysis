#!/usr/bin/env python3
"""
Universal CPM normalisation + replicate averaging + plotting for genomecov -d tables.

Input : a folder of 3-column tables (reference, position, depth) from bedtools genomecov -d
        + samples.csv with columns:  file, group, role, replicate, total_reads
          group = comparison name (e.g. HEK_JEV), role = infected or mock
          total_reads = 'total primary' from samtools flagstat
Usage : python normalise_and_plot.py samples.csv --folder tables --tag coverage \
            [--group HEK_JEV] [--zoom 10395 10964] [--annot annotations.csv] [--ymax 120]
        --group limits the run to one or more groups (needed when groups use different zoom windows)
annotations.csv (optional): name,start,end   (e.g. SL-II,10400,10460)
Outputs (per group): <group>_<tag>_CPM.csv, .png and .svg
"""
import argparse, os
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_settings(path):
    """Read a settings csv saved by VS Code or by (Russian) Excel: handles , or ; and a BOM."""
    first = open(path, encoding="utf-8-sig").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")

ap = argparse.ArgumentParser()
ap.add_argument("samples"); ap.add_argument("--folder", default=".")
ap.add_argument("--tag", default="coverage", help="label for outputs, e.g. coverage, 5prime, 3prime")
ap.add_argument("--zoom", nargs=2, type=int); ap.add_argument("--annot"); ap.add_argument("--ymax", type=float)
ap.add_argument("--out", default="results")
ap.add_argument("--group", nargs="+", help="only process these group(s), e.g. --group HEK_JEV  (default: all groups in samples.csv)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
S = read_settings(a.samples)
if a.group:
    missing = [g for g in a.group if g not in set(S["group"])]
    if missing:
        raise SystemExit(f"Group(s) not found in {a.samples}: {missing}. Available: {sorted(set(S['group']))}")
    S = S[S["group"].isin(a.group)]
annot = read_settings(a.annot) if a.annot else None
colour = {"infected": "tab:blue", "mock": "tab:gray"}

for group, G in S.groupby("group", sort=False):
    out = None; stats = {}
    for role, R in G.groupby("role", sort=False):
        cols = []
        for _, r in R.iterrows():
            df = pd.read_csv(os.path.join(a.folder, r["file"]), sep="\t", header=None, names=["ref", "position", "depth"])
            if out is None:
                out = pd.DataFrame({"position": df["position"]}); first_file, first_ref = r["file"], df["ref"].iloc[0]
            elif len(df) != len(out) or df["ref"].iloc[0] != first_ref:
                raise SystemExit(f"\nERROR in group '{group}': {r['file']} has {len(df)} rows and reference '{df['ref'].iloc[0]}', "
                                 f"but {first_file} has {len(out)} rows and reference '{first_ref}'.\n"
                                 f"Tables in one group must come from the SAME virus reference. Check that {r['file']} is the right table.")
            c = f"{role}_rep{r['replicate']}_CPM"
            out[c] = df["depth"].values / r["total_reads"] * 1e6   # CPM
            cols.append(c)
        out[f"{role}_mean_CPM"] = out[cols].mean(axis=1)
        out[f"{role}_SD"] = out[cols].std(axis=1)          # SD across replicates
        stats[role] = (cols, len(cols))
    out.to_csv(os.path.join(a.out, f"{group}_{a.tag}_CPM.csv"), index=False)

    windows = [("Full genome", out["position"].min(), out["position"].max())]
    if a.zoom: windows.append((f"Zoom {a.zoom[0]}-{a.zoom[1]}", *a.zoom))
    fig, axes = plt.subplots(len(windows), 1, figsize=(12, 3.8 * len(windows)), squeeze=False)
    for ax, (title, lo, hi) in zip(axes[:, 0], windows):
        m = out[(out.position >= lo) & (out.position <= hi)]
        for role, (cols, n) in stats.items():
            for c in cols: ax.plot(m.position, m[c], lw=0.4, color=colour.get(role, "k"), alpha=0.4)   # each replicate, faint
            ax.plot(m.position, m[f"{role}_mean_CPM"], lw=1.2, color=colour.get(role, "k"), label=f"{role} mean (n={n})")
            ax.fill_between(m.position, (m[f"{role}_mean_CPM"] - m[f"{role}_SD"]).clip(lower=0),
                            m[f"{role}_mean_CPM"] + m[f"{role}_SD"], color=colour.get(role, "k"), alpha=0.15)
        if annot is not None:
            for _, r in annot.iterrows():
                if r["end"] >= lo and r["start"] <= hi:
                    ax.axvspan(r["start"], r["end"], color="gold", alpha=0.25)
                    ax.text((r["start"] + r["end"]) / 2, ax.get_ylim()[1] * 0.97, r["name"], ha="center", va="top", fontsize=7)
        if a.ymax: ax.set_ylim(0, a.ymax)
        ax.set_title(f"{group} - {title} ({a.tag})"); ax.set_ylabel("CPM"); ax.legend(loc="upper left")
    axes[-1, 0].set_xlabel("Genome position (nt, 1-based)")
    plt.tight_layout()
    for ext in ("png", "svg"): plt.savefig(os.path.join(a.out, f"{group}_{a.tag}.{ext}"), dpi=200)
    plt.close(); print("done:", group)
