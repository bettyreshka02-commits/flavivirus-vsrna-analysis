#!/usr/bin/env python3
"""
Thesis-style figure for one group: coverage + 5' ends + 3' ends + fragment map, stacked on one x-axis,
with the top peaks labelled (position and nucleotide) and optional structural elements shaded.

Needs (all made by the earlier scripts, run from your project folder):
    results/<group>_coverage_CPM.csv   results/<group>_5prime_CPM.csv   results/<group>_3prime_CPM.csv
    peaks/<group>_5prime_peaks.csv     peaks/<group>_3prime_peaks.csv   peaks/<group>_fragment_pairs.csv

Usage:
    python plot_peaks_figure.py --group HEK_KUNV --region 10399 11015
    python plot_peaks_figure.py --group HEK_JEV  --region 10395 10964 --annot elements_JEV.csv

Options:
    --annot FILE       csv with columns name,start,end (e.g. SL-I,10399,10450). Shaded on every panel.
    --label_top N      label the N strongest peaks in each end track (default 8)
    --frag_top K       draw the K best fragment pairs (default 12)
    --frag_max_len L   ignore pairs longer than L nt (default 120)
    --ymax_cov / --ymax_ends   fix the y-axis limits (useful to compare groups)
Output: figures/<group>_peaks_figure.png and .svg
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_COV, C_5, C_3, C_MOCK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"   # validated categorical slots 1-3 + neutral


def read_table(path):
    first = open(path, encoding="utf-8-sig").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"Cannot find {path}. Run the earlier script that creates it first.")
    return read_table(path)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def label_peaks(ax, peaks, colour, genome_end, lo, hi, n, ymax, min_gap):
    """Label the strongest n peaks; skip labels that would sit on top of a stronger neighbour."""
    p = peaks[(peaks.position >= lo) & (peaks.position <= hi)].sort_values("mean_CPM", ascending=False)
    done, note = [], False
    for _, r in p.iterrows():
        if len(done) >= n:
            break
        if any(abs(r.position - d) < min_gap for d in done):
            continue
        end_flag = ""
        if int(r.position) == genome_end and r.end_type == "3prime":
            end_flag, note = "†", True               # genome end: not a cleavage site
        ax.text(r.position, r.mean_CPM + ymax * 0.02, f"{int(r.position)} {r.nucleotide}{end_flag}",
                rotation=90, ha="center", va="bottom", fontsize=7, color=INK)
        done.append(r.position)
    return note


def pack_rows(frags, gap):
    rows, ends = [], []
    for _, f in frags.sort_values("start_5prime_pos").iterrows():
        for i, e in enumerate(ends):
            if f.start_5prime_pos > e + gap:
                ends[i] = f.end_3prime_pos; rows.append(i); break
        else:
            ends.append(f.end_3prime_pos); rows.append(len(ends) - 1)
    out = frags.sort_values("start_5prime_pos").copy(); out["row"] = rows
    return out, len(ends)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--region", nargs=2, type=int, help="window to plot, e.g. 10399 11015 (3' UTR)")
    ap.add_argument("--results", default="results"); ap.add_argument("--peaks", default="peaks")
    ap.add_argument("--out", default="figures"); ap.add_argument("--annot")
    ap.add_argument("--label_top", type=int, default=8); ap.add_argument("--frag_top", type=int, default=12)
    ap.add_argument("--frag_max_len", type=int, default=120)
    ap.add_argument("--ymax_cov", type=float); ap.add_argument("--ymax_ends", type=float)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    cov = need(os.path.join(a.results, f"{a.group}_coverage_CPM.csv"))
    e5 = need(os.path.join(a.results, f"{a.group}_5prime_CPM.csv"))
    e3 = need(os.path.join(a.results, f"{a.group}_3prime_CPM.csv"))
    p5 = need(os.path.join(a.peaks, f"{a.group}_5prime_peaks.csv"))
    p3 = need(os.path.join(a.peaks, f"{a.group}_3prime_peaks.csv"))
    fr = need(os.path.join(a.peaks, f"{a.group}_fragment_pairs.csv"))
    genome_end = int(cov.position.max())
    lo, hi = a.region if a.region else (int(cov.position.min()), genome_end)
    annot = read_table(a.annot) if a.annot else None
    min_gap = max(3, (hi - lo) / 90)

    fr = fr[(fr.fragment_length_nt <= a.frag_max_len) & (fr.start_5prime_pos >= lo) & (fr.end_3prime_pos <= hi)]
    fr = fr.sort_values("pair_score_min_CPM", ascending=False).head(a.frag_top)
    fr, nrows = pack_rows(fr, gap=max(2, (hi - lo) / 150)) if len(fr) else (fr, 1)

    fig, axs = plt.subplots(4, 1, figsize=(12.5, 10), sharex=True,
                            gridspec_kw={"height_ratios": [2.3, 1.7, 1.7, 0.4 * max(nrows, 2) + 0.5], "hspace": 0.12})
    m = lambda d: d[(d.position >= lo) & (d.position <= hi)]

    # 1 coverage
    ax = axs[0]; style(ax); d = m(cov)
    ax.plot(d.position, d.infected_mean_CPM, color=C_COV, lw=1.4, label="infected, mean of replicates")
    ax.fill_between(d.position, (d.infected_mean_CPM - d.infected_SD).clip(lower=0), d.infected_mean_CPM + d.infected_SD,
                    color=C_COV, alpha=0.15, lw=0, label="± 1 SD")
    ax.plot(d.position, d.mock_mean_CPM, color=C_MOCK, lw=1.2, label="mock")
    ax.set_ylabel("coverage (CPM)", color=INK2, fontsize=9)
    if a.ymax_cov: ax.set_ylim(0, a.ymax_cov)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"{a.group}  –  positions {lo}–{hi}", loc="left", fontsize=11, color=INK, pad=(20 if a.annot else 6))

    # 2 and 3: end tracks as stems
    note = False
    for ax, tab, pk, col, name in ((axs[1], e5, p5, C_5, "5′ ends (fragment starts)"),
                                   (axs[2], e3, p3, C_3, "3′ ends (fragment ends)")):
        style(ax); d = m(tab); d = d[d.infected_mean_CPM > 0]
        ax.vlines(d.position, 0, d.infected_mean_CPM, color=col, lw=1.1)
        ymax = a.ymax_ends or float(m(tab).infected_mean_CPM.max()) * 1.0
        ax.set_ylim(0, ymax * 1.38)
        ax.set_ylabel(f"{name}\n(CPM)", color=INK2, fontsize=9)
        pk = pk.copy(); pk["end_type"] = "5prime" if "5′" in name else "3prime"
        note |= label_peaks(ax, pk, col, genome_end, lo, hi, a.label_top, ymax, min_gap)

    # 4 fragment map
    ax = axs[3]; style(ax); ax.grid(False)
    for _, f in fr.iterrows():
        y = f.row
        ax.add_patch(plt.Rectangle((f.start_5prime_pos, y + 0.15), f.end_3prime_pos - f.start_5prime_pos + 1, 0.55,
                                   color=INK2, alpha=0.55, lw=0))
        ax.add_patch(plt.Rectangle((f.start_5prime_pos, y + 0.15), max(1.5, (hi - lo) / 400), 0.55, color=C_5, lw=0))
        ax.add_patch(plt.Rectangle((f.end_3prime_pos + 1 - max(1.5, (hi - lo) / 400), y + 0.15), max(1.5, (hi - lo) / 400), 0.55, color=C_3, lw=0))
        ax.text(f.end_3prime_pos + (hi - lo) / 200, y + 0.42, f"{int(f.fragment_length_nt)} nt", fontsize=7, color=INK, va="center")
    ax.set_ylim(0, max(nrows, 1) + 0.2); ax.set_yticks([])
    ax.set_ylabel(f"top {len(fr)} candidate\nfragments", color=INK2, fontsize=9)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("genome position (nt, 1-based)", color=INK2, fontsize=9)

    # shared: structural elements
    ax.set_xlim(lo - (hi - lo) * 0.005, hi + (hi - lo) * 0.02)
    if annot is not None:
        for _, r in annot.iterrows():
            if r["end"] >= lo and r["start"] <= hi:
                for k, axx in enumerate(axs):
                    axx.axvspan(r["start"], r["end"], color="#eda100", alpha=0.13, lw=0)
                axs[0].text((max(r["start"], lo) + min(r["end"], hi)) / 2, 1.01, r["name"], transform=axs[0].get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=8, color=INK)

    foot = "orange = fragment start (5′ end), green = fragment end (3′ end). Candidate fragments pair a 5′ peak with a 3′ peak downstream; they are not confirmed molecules."
    if note: foot += "  † 3′ end at the last genome nucleotide: not a cleavage site."
    fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK2, ha="left")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.07)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(a.out, f"{a.group}_peaks_figure.{ext}"), dpi=200)
    print(f"Saved {a.out}/{a.group}_peaks_figure.png and .svg")


if __name__ == "__main__":
    main()
