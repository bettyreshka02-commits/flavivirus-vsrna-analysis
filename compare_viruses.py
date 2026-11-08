#!/usr/bin/env python3
"""
Three viruses side by side: coverage and the strongest discrete small RNAs of each 3' UTR, drawn on ONE shared axis.

How it works
  - The 3' UTR of every virus is aligned to the 3' UTR of the first virus in viruses.csv (the reference axis).
    Every position of KUNV / NY99 / JEV is converted into a reference-axis position, so small RNAs from different viruses
    can be compared at equivalent places.
  - For each virus the plot shows infected coverage (CPM) and the top discrete small RNAs (sense strand, infection-specific,
    seen in >= --min_reps replicates). A diamond marks an RNA that is also found at the same place (+/- --tol nt) in another virus.
  - Ragged-end variants of one species (start and end both within --merge nt, default 2) are merged into one small RNA; its CPM is the
    sum of the variants and its label shows the most abundant length with the range in brackets.
  - Also writes cross_virus_srnas.csv with positions in the virus's own numbering AND in reference-axis numbering.

Needs, from earlier steps:
    results/<group>_coverage_CPM.csv                  (normalise_and_plot.py, tag coverage)
    srna_results/<group>_unique_srna_loci.csv         (length_and_unique_srnas.py)
and a small file viruses.csv  (one row per virus; the FIRST row is the reference axis):
    name,group,fasta,utr_start,utr_end
    KUNV,HEK_KUNV,reference files/WNV_KUN_NSW2011.fa,10399,11015
    NY99,HEK_NY99,reference files/WNV_NY99.fa,10399,11029
    JEV,HEK_JEV,reference files/JEV_FU.fa,10395,10964

Usage:
    python compare_viruses.py viruses.csv [--annot elements.csv] [--top 12] [--tol 3] [--merge 2]
elements.csv (optional, REFERENCE-virus numbering): name,start,end   e.g. DB1,10700,10760
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_COV, C_5, C_3, C_BAR = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"


def read_settings(path):
    first = open(path, encoding="utf-8-sig").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def read_fasta(path):
    seq, n = [], 0
    for line in open(path):
        if line.startswith(">"):
            n += 1
            if n > 1:
                break
        else:
            seq.append(line.strip().upper().replace("U", "T"))
    return "".join(seq)


def align_semiglobal(a, b, match=2, mismatch=-2, gopen=-5, gext=-1):
    """Gotoh alignment with free end gaps. Returns list of aligned (i, j) pairs, 1-based positions in a and b."""
    n, m, NEG = len(a), len(b), -10 ** 6
    A, B = np.frombuffer(a.encode(), dtype=np.uint8), np.frombuffer(b.encode(), dtype=np.uint8)
    M = np.full((n + 1, m + 1), NEG, dtype=np.int64); X = M.copy(); Y = M.copy()
    M[:, 0] = 0; M[0, :] = 0
    for i in range(1, n + 1):
        sc = np.where(B == A[i - 1], match, mismatch)
        M[i, 1:] = np.maximum(np.maximum(M[i - 1, :-1], X[i - 1, :-1]), Y[i - 1, :-1]) + sc
        X[i, 1:] = np.maximum(np.maximum(M[i - 1, 1:] + gopen, X[i - 1, 1:] + gext), Y[i - 1, 1:] + gopen)
        mi, xi, yi = M[i].tolist(), X[i].tolist(), Y[i].tolist()
        for j in range(1, m + 1):
            yi[j] = max(mi[j - 1] + gopen, yi[j - 1] + gext, xi[j - 1] + gopen)
        Y[i] = yi
    last_row, last_col = M[n, 1:], M[1:, m]
    if last_row.max() >= last_col.max():
        i, j = n, int(last_row.argmax()) + 1
    else:
        i, j = int(last_col.argmax()) + 1, m
    pairs, st = [], "M"
    while i > 0 and j > 0:
        if st == "M":
            pairs.append((i, j))
            cur = M[i, j] - (match if A[i - 1] == B[j - 1] else mismatch)
            i, j = i - 1, j - 1
            if i == 0 or j == 0:
                break
            st = "M" if M[i, j] == cur else ("X" if X[i, j] == cur else "Y")
        elif st == "X":
            cur = X[i, j]; i -= 1
            if i == 0: break
            st = "M" if M[i, j] + gopen == cur else ("X" if X[i, j] + gext == cur else "Y")
        else:
            cur = Y[i, j]; j -= 1
            if j == 0: break
            st = "M" if M[i, j] + gopen == cur else ("Y" if Y[i, j] + gext == cur else "X")
    return pairs[::-1]


def map_to_reference(ref_utr, utr):
    """For each position (1-based) of utr: the equivalent position of ref_utr (insertions share the previous position)."""
    pairs = align_semiglobal(ref_utr, utr)
    if not pairs:
        raise SystemExit("The 3' UTRs could not be aligned. Check the UTR coordinates in viruses.csv.")
    mp = np.zeros(len(utr) + 1, dtype=int)
    for i, j in pairs:
        mp[j] = i
    cur = pairs[0][0]
    for j in range(1, len(utr) + 1):
        if mp[j] == 0: mp[j] = cur
        else: cur = mp[j]
    ident = 100 * sum(ref_utr[i - 1] == utr[j - 1] for i, j in pairs) / len(pairs)
    return mp, ident, pairs


def cluster_loci(L, merge):
    """Merge ragged-end variants: loci whose start AND end are both within +/- merge nt of a stronger locus join it.
    The strongest member is the representative; CPM of the cluster is the sum of its members (they are different reads)."""
    L = L.sort_values("infected_mean_CPM", ascending=False)
    reps, members = [], []
    for r in L.itertuples(index=False):
        for k, c in enumerate(reps):
            if abs(r.start - c.start) <= merge and abs(r.end - c.end) <= merge:
                members[k].append(r); break
        else:
            reps.append(r); members.append([r])
    out = L.iloc[0:0].copy() if len(L) == 0 else pd.DataFrame([r._asdict() for r in reps])
    if len(out):
        out["cluster_CPM"] = [sum(m.infected_mean_CPM for m in g) for g in members]
        out["n_variants"] = [len(g) for g in members]
        out["len_min"] = [min(m.length_nt for m in g) for g in members]
        out["len_max"] = [max(m.length_nt for m in g) for g in members]
        out["ref_start"] = [g[0].ref_start for g in members]; out["ref_end"] = [g[0].ref_end for g in members]
    return out


def label_of(f, mark):
    rng = f"  ({int(f.len_min)}–{int(f.len_max)})" if f.len_max > f.len_min else ""
    return f"{mark}{int(f.length_nt)} nt{rng}"


def pack_rows(df, pad):
    """Greedy rows: a small RNA goes in the first row where its bar AND its text label (extending to the right) are clear."""
    ends, rows = [], []
    d = df.sort_values("ref_start").copy()
    for _, f in d.iterrows():
        right = f.ref_end + pad(label_of(f, "◆ "))
        for k, e in enumerate(ends):
            if f.ref_start > e:
                ends[k] = right; rows.append(k); break
        else:
            ends.append(right); rows.append(len(ends) - 1)
    d["row"] = rows
    return d, max(len(ends), 1)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("viruses"); ap.add_argument("--results", default="results"); ap.add_argument("--srna", default="srna_results")
    ap.add_argument("--annot"); ap.add_argument("--top", type=int, default=12); ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--min_reps", type=int, default=2); ap.add_argument("--mock_fraction", type=float, default=0.1)
    ap.add_argument("--min_cpm", type=float, default=0, help="ignore small RNAs below this mean CPM")
    ap.add_argument("--merge", type=int, default=2, help="merge ragged-end variants whose start and end differ by <= this many nt (0 = off)")
    ap.add_argument("--out", default="srna_results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    V = read_settings(a.viruses)
    ref = V.iloc[0]
    ref_seq = read_fasta(ref["fasta"])[int(ref["utr_start"]) - 1:int(ref["utr_end"])]
    ref_start, ref_len = int(ref["utr_start"]), len(ref_seq)

    info = []
    for _, v in V.iterrows():
        s0, e0 = int(v["utr_start"]), int(v["utr_end"])
        seq = read_fasta(v["fasta"])
        utr = seq[s0 - 1:e0]
        mp, ident, _ = map_to_reference(ref_seq, utr)
        cov = pd.read_csv(os.path.join(a.results, f"{v['group']}_coverage_CPM.csv"))
        loci = pd.read_csv(os.path.join(a.srna, f"{v['group']}_unique_srna_loci.csv"))
        loci = loci[(loci.strand == "sense") & (loci.start >= s0) & (loci.end <= e0)].copy()
        loci = loci[(loci.infected_replicates_detected >= a.min_reps) & (loci.infected_mean_CPM >= a.min_cpm)]
        if "mock" in loci.filter(like="_mean_CPM").columns.str.split("_").str[0].tolist():
            loci = loci[loci.mock_mean_CPM <= a.mock_fraction * loci.infected_mean_CPM]
        to_ref = lambda p: ref_start + int(mp[p - s0 + 1]) - 1
        loci["ref_start"] = loci.start.map(to_ref); loci["ref_end"] = loci.end.map(to_ref)
        loci["virus"] = v["name"]
        n_before = len(loci)
        loci = cluster_loci(loci, a.merge)
        info.append(dict(name=v["name"], group=v["group"], s0=s0, e0=e0, mp=mp, ident=ident, cov=cov, loci=loci, len=len(utr)))
        print(f"{v['name']}: 3' UTR {len(utr)} nt, identity to {ref['name']} over aligned columns = {ident:.0f}%, {n_before} candidate loci -> {len(loci)} distinct small RNAs after merging ragged ends (±{a.merge} nt)")

    # conservation: same place (+/- tol) in another virus
    for x in info:
        seen = []
        for _, r in x["loci"].iterrows():
            hits = []
            for y in info:
                if y is x: continue
                m = y["loci"][(abs(y["loci"].ref_start - r.ref_start) <= a.tol) & (abs(y["loci"].ref_end - r.ref_end) <= a.tol)]
                hits += [f"{y['name']}:{h}" for h in m.locus_id]
            seen.append(", ".join(hits))
        x["loci"]["also_found_in"] = seen
    allc = pd.concat([x["loci"].nlargest(a.top, "cluster_CPM") for x in info], ignore_index=True)
    cols = ["virus", "locus_id", "start", "end", "ref_start", "ref_end", "length_nt", "len_min", "len_max", "n_variants",
            "infected_mean_CPM", "cluster_CPM", "infected_replicates_detected", "also_found_in"]
    allc[cols].round(2).to_csv(os.path.join(a.out, "cross_virus_srnas.csv"), index=False)

    # ---------- figure ----------
    tops, ratios = [], []
    xmin, xmax = ref_start, ref_start + ref_len - 1
    nt_per_char = 0.062 / 11.3 * ref_len * 1.03          # width of one 7-pt character in nt on the shared axis
    pad = lambda text: (len(text) + 2) * nt_per_char + ref_len / 200
    for x in info:
        t, nrows = pack_rows(x["loci"].nlargest(a.top, "cluster_CPM"), pad)
        tops.append((t, nrows)); ratios += [2.0, 0.38 * nrows + 0.55]
    fig, axs = plt.subplots(2 * len(info), 1, figsize=(12.5, 3.3 * len(info) + 1.2), sharex=True,
                            gridspec_kw={"height_ratios": ratios, "hspace": 0.34})
    annot = read_settings(a.annot) if a.annot else None
    for k, x in enumerate(info):
        axc, axb = axs[2 * k], axs[2 * k + 1]
        style(axc); style(axb); axb.grid(False)
        cov = x["cov"]; cov = cov[(cov.position >= x["s0"]) & (cov.position <= x["e0"])]
        xr = ref_start + x["mp"][(cov.position - x["s0"] + 1).values] - 1
        d = pd.DataFrame({"x": xr, "y": cov.infected_mean_CPM.values, "sd": cov.infected_SD.values}).groupby("x").max()
        full = d.reindex(np.arange(xmin, xmax + 1))
        axc.fill_between(full.index, (full.y - full.sd).clip(lower=0), full.y + full.sd, color=C_COV, alpha=0.15, lw=0)
        axc.plot(full.index, full.y, color=C_COV, lw=1.3)
        axc.set_ylabel("coverage (CPM)", fontsize=8.5, color=INK2); axc.set_ylim(bottom=0)
        ttl = f"{x['name']} ({x['group']}), 3′ UTR {x['e0'] - x['s0'] + 1} nt"
        if k > 0: ttl += f"  –  {x['ident']:.0f}% identity to {info[0]['name']} over aligned positions"
        axc.set_title(ttl, loc="left", fontsize=10, color=INK)
        t, nrows = tops[k]
        for _, f in t.iterrows():
            y = f.row; w = max(1.5, ref_len / 400)
            axb.add_patch(plt.Rectangle((f.ref_start, y + 0.12), f.ref_end - f.ref_start + 1, 0.56, color=C_BAR, alpha=0.55, lw=0))
            axb.add_patch(plt.Rectangle((f.ref_start, y + 0.12), w, 0.56, color=C_5, lw=0))
            axb.add_patch(plt.Rectangle((f.ref_end + 1 - w, y + 0.12), w, 0.56, color=C_3, lw=0))
            txt = label_of(f, "◆ " if f.also_found_in else "")
            if f.ref_end + pad(txt) > xmax + ref_len * 0.03:          # no room on the right: put the label left of the bar
                axb.text(f.ref_start - ref_len / 200, y + 0.40, txt, fontsize=7, color=INK, va="center", ha="right")
            else:
                axb.text(f.ref_end + ref_len / 200, y + 0.40, txt, fontsize=7, color=INK, va="center")
        axb.set_ylim(0, nrows + 0.1); axb.set_yticks([]); axb.spines["left"].set_visible(False)
        axb.set_ylabel(f"top {len(t)} small RNAs", fontsize=8.5, color=INK2)
        if annot is not None:
            for _, r in annot.iterrows():
                for ax in (axc, axb):
                    ax.axvspan(r["start"], r["end"], color="#eda100", alpha=0.13, lw=0)
                if k == 0:
                    axc.text((r["start"] + r["end"]) / 2, 1.01, r["name"], transform=axc.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=INK)
    axs[-1].set_xlim(xmin - ref_len * 0.005, xmax + ref_len * 0.03)
    axs[-1].set_xlabel(f"position on the {info[0]['name']} 3′ UTR axis (genome numbering of {info[0]['name']}); other viruses are aligned to it", fontsize=9, color=INK2)
    foot = ("orange = 5′ end, green = 3′ end of the small RNA. ◆ = also found at the same place (±" + str(a.tol) +
            " nt) in another virus.\nTop small RNAs are sense, infection-specific and seen in >= " + str(a.min_reps) +
            " replicates. Ragged-end variants (±" + str(a.merge) + " nt) are merged; label = most abundant length (range of lengths).")
    fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK2)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.085)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(a.out, f"cross_virus_comparison.{ext}"), dpi=200)
    print(f"\nSaved {a.out}/cross_virus_comparison.png/.svg and cross_virus_srnas.csv")


if __name__ == "__main__":
    main()
