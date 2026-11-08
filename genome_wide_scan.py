#!/usr/bin/env python3
"""
Genome-wide end-peak scan (Results 3.5).
Run from the kun_a folder:   python genome_wide_scan.py
Needs: find_peaks_and_fragments.py, samples.csv, results/<group>_{5prime,3prime,coverage}_CPM.csv, reference files/*.fa
Writes: peaks_genome_wide/<group>_{5prime,3prime}_peaks.csv, genome_wide_summary.csv (Table 3.5), figures/genome_wide_peaks.png
Peaks use the same PASS filter as find_peaks_and_fragments.py (2 of 3 replicates, >=10 reads, mock <10%), with no limit on peak number.
"""
import importlib.util, os, sys, types
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = sys.argv[1] if len(sys.argv) > 1 else "."
spec = importlib.util.spec_from_file_location("fp", os.path.join(base, "find_peaks_and_fragments.py"))
fp = importlib.util.module_from_spec(spec); spec.loader.exec_module(fp)
S = fp.read_settings(os.path.join(base, "samples.csv"))
# group: (fasta, 3' UTR start, 3' UTR end). A549 positions below 10,455 are the same in NSW2011 and FLSDX.
CFG = {"HEK_KUNV": ("WNV_KUN_NSW2011.fa", 10399, 11015), "A549_KUNV": ("WNV_KUN_NSW2011.fa", 10399, 11015),
       "HEK_JEV": ("JEV_FU.fa", 10395, 10964), "HEK_NY99": ("WNV_NY99.fa", 10399, 11029)}
LAB = {"HEK_KUNV": "HEK293 KUNV", "A549_KUNV": "A549 KUNV", "HEK_JEV": "HEK293 JEV", "HEK_NY99": "HEK293 NY99"}
a = types.SimpleNamespace(window=3, min_cpm=5, region=None, top=100000, min_rep_cpm=2, min_reps=2, min_reads=10, mock_fraction=0.1)
os.makedirs(os.path.join(base, "peaks_genome_wide"), exist_ok=True); os.makedirs(os.path.join(base, "figures"), exist_ok=True)

peaks, rows = {}, []
for g, (fa, u0, u1) in CFG.items():
    _, seq = fp.read_fasta(os.path.join(base, "reference files", fa))
    tot = S[(S.group == g) & (S.role == "infected")].sort_values("replicate").total_reads.astype(float).tolist()
    p = {}
    for tag in ("5prime", "3prime"):
        df = pd.read_csv(os.path.join(base, "results", f"{g}_{tag}_CPM.csv"))
        assert len(df) == len(seq), (g, len(df), len(seq))
        p[tag] = fp.call_peaks(df, tag, seq, tot, a)
        p[tag].to_csv(os.path.join(base, "peaks_genome_wide", f"{g}_{tag}_peaks.csv"), index=False)
    peaks[g] = p
    P = pd.concat(p.values()); P = P[P.PASS]
    o, i = P[P.position < u0], P[P.position >= u0]
    rows.append({"condition": LAB[g], "outside_peaks": len(o), "outside_per_kb": round(len(o) / ((u0 - 1) / 1000)),
                 "outside_median_CPM": round(o.mean_CPM.median(), 1), "outside_over_100": int((o.mean_CPM > 100).sum()),
                 "outside_strongest": round(o.mean_CPM.max()), "inside_peaks": len(i),
                 "inside_per_kb": round(len(i) / ((u1 - u0 + 1) / 1000)), "inside_median_CPM": round(i.mean_CPM.median(), 1),
                 "inside_over_100": int((i.mean_CPM > 100).sum()), "inside_strongest": round(i.mean_CPM.max())})
summ = pd.DataFrame(rows); summ.to_csv(os.path.join(base, "genome_wide_summary.csv"), index=False); print(summ.to_string(index=False))

# reproducibility between cell lines: strongest A549 peaks outside the 3' UTR found in HEK293 (same end, +/-1 nt) vs shuffles
def passing(g):
    P = pd.concat(peaks[g].values()); return P[P.PASS & (P.position < CFG[g][1])]
def n_found(A, B, topn):
    a_ = A.nlargest(topn, "mean_CPM"); return sum(((B.end_type == r.end_type) & ((B.position - r.position).abs() <= 1)).any() for r in a_.itertuples())
A, H = passing("A549_KUNV"), passing("HEK_KUNV"); rng = np.random.default_rng(1)
for n in (30, 60, 100):
    chance = np.mean([n_found(A, H.assign(position=rng.integers(1, 10398, len(H))), n) for _ in range(200)])
    print(f"top {n} A549 peaks also in HEK293: {n_found(A, H, n)} (chance {chance:.1f})")

# figure (y axis limit in CPM; taller peaks are cut off and labelled with their height)
YL = 550
C5, C3 = "#1b6ca8", "#c8590b"
fig, axs = plt.subplots(4, 1, figsize=(10, 9.5), sharey=True)
for ax, (g, (fa, u0, u1)) in zip(axs, CFG.items()):
    ax.axvspan(u0, u1, color="#d9d9d9", lw=0)
    for tag, c, s in (("5prime", C5, 1), ("3prime", C3, -1)):
        P = peaks[g][tag]; P = P[P.PASS]; ax.vlines(P.position, 0, s * P.mean_CPM, color=c, lw=0.7)
    ax.axhline(0, color="#555", lw=.6); ax.set_xlim(0, u1 + 100); ax.set_ylabel("end peak (CPM)")
    r = summ[summ.condition == LAB[g]].iloc[0]
    ax.set_title(f"{LAB[g]}: {r.outside_peaks} passing peaks outside the 3′ UTR, {r.outside_over_100} over 100 CPM", loc="left", fontsize=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.text((u0 + u1) / 2, 1.02, "3′ UTR", ha="center", va="bottom", fontsize=8, color="#444", transform=ax.get_xaxis_transform())
    for tag, sgn in (("5prime", 1), ("3prime", -1)):          # label peaks that are cut off by the axis limit
        P = peaks[g][tag]; P = P[P.PASS & (P.mean_CPM > YL)]
        for r_ in P.itertuples():
            ax.annotate(f"{r_.mean_CPM:,.0f}", (r_.position, sgn * YL), xytext=(-4, -9 * sgn - 3), textcoords="offset points", ha="right", fontsize=7, color="#333")
axs[0].set_ylim(-YL, YL); axs[-1].set_xlabel("Genome position (nt)")
axs[0].legend(handles=[plt.Line2D([], [], color=C5, lw=2, label="5′ ends (up)"), plt.Line2D([], [], color=C3, lw=2, label="3′ ends (down)")], frameon=False, loc="upper left", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(base, "figures", "genome_wide_peaks.png"), dpi=200)
