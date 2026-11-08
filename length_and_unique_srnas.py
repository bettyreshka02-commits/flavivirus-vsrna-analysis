#!/usr/bin/env python3
"""
Read-length plots + collapsed list of unique small RNAs, for one group (e.g. HEK_KUNV).

What it does
  1. Read-length plot: how many aligned reads of each length (X = length in nt, Y = reads), sense and antisense,
     infected (mean of replicates +/- SD) next to mock.
  2. Collapse reads: identical reads (same strand, same start/end, same sequence) are merged into one
     "unique small RNA" with its read count in every replicate. This is the same idea as miRDeep2's collapse/map
     step. Collapsed input (counts in the read names) and plain aligned reads are both accepted.
  3. Writes a ranked table of unique small RNAs (and one of discrete loci = same start and end).

Input (made in Galaxy, see the instructions): one small table per sample with the ALIGNED reads only. Three layouts work:
  (a) COLLAPSED reads mapped back to the genome (supervisor route):  QNAME, FLAG, reference, position, CIGAR, sequence
      The read count is taken from the collapsed read name (FASTX 'Collapse sequences': 12-3354, miRDeep2: abc_12_x3354).
  (b) uncollapsed aligned reads:  FLAG, reference, position, CIGAR, sequence      (every line = 1 read)
  (c) a full SAM file (names with a count are handled as in (a)).
  Files must have the same names as the 'file' column of samples.csv (the extension may differ).

samples.csv columns: file, group, role (infected/mock), replicate, total_reads   (same file as before)

Usage (run in your project folder):
    python length_and_unique_srnas.py samples.csv --folder aligned_tables --group HEK_KUNV --region 10399 11015

Options:
    --region START END   3' UTR window: adds a second row of plots and an 'in_region' column
    --min_len / --max_len  range of the length plot (default 15 to 60)
    --min_total N        keep unique RNAs with at least N reads summed over infected replicates (default 10)
    --mock_fraction F    'infection_specific' = mock CPM <= F x infected CPM (default 0.1)
    --min_read_len N     ignore reads shorter than N nt (default 18; very short reads align by chance)
    --raw                plot read counts instead of CPM
Outputs (folder 'srna_results'): <group>_length_distribution.csv/.png/.svg, <group>_unique_srnas.csv/.xlsx
"""
import argparse, collections, os, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_INF, C_MOCK, INK, INK2, GRID = "#2a78d6", "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def read_settings(path):
    first = open(path, encoding="utf-8-sig").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def find_file(folder, name):
    p = os.path.join(folder, name)
    if os.path.exists(p):
        return p
    if os.path.isdir(folder):
        stem = os.path.splitext(name)[0]
        for f in os.listdir(folder):
            if os.path.splitext(f)[0] == stem:
                return os.path.join(folder, f)
    raise SystemExit(f"\nCannot find {name} in folder '{folder}'. The file names must match the 'file' column of samples.csv.")


def revcomp(seq):
    return seq.upper().translate(str.maketrans("ACGTUN", "TGCAAN"))[::-1]


COUNT = re.compile(r"^(?:\d+-(\d+)|[A-Za-z]+_\d+_x(\d+))$")     # 12-3354 (FASTX collapser) or abc_12_x3354 (miRDeep2)


def read_count(qname):
    m = COUNT.match(qname)
    return int(m.group(1) or m.group(2)) if m else 1


def load_sample(path, min_read_len=0):
    """Counter of (strand, start, end, sequence-on-genome) -> number of reads (collapsed counts are added up)."""
    reads, lines, collapsed, dropped = collections.Counter(), 0, False, 0
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            if not line.strip() or line[0] == "@":
                continue
            p = line.rstrip("\r\n").split("\t")
            n = len(p)
            try:
                if n >= 10:   q, flag, pos, cigar, seq = p[0], int(p[1]), int(p[3]), p[5], p[9]    # full SAM
                elif n == 6:  q, flag, pos, cigar, seq = p[0], int(p[1]), int(p[3]), p[4], p[5]    # QNAME, FLAG, ref, POS, CIGAR, SEQ
                elif n == 5:  q, flag, pos, cigar, seq = "", int(p[0]), int(p[2]), p[3], p[4]
                elif n == 4:  q, flag, pos, cigar, seq = "", int(p[0]), int(p[2]), "*", p[3]
                else:         continue
            except ValueError:
                continue                                                                      # header or junk line
            if flag & (4 | 256 | 2048):                                                       # unmapped / secondary / supplementary
                continue
            c = read_count(q)
            collapsed |= c > 1
            if len(seq) < min_read_len:                                                       # too short to be a real small RNA
                dropped += c
                continue
            lines += 1
            span = sum(int(a) for a, op in CIG.findall(cigar) if op in "MDN=X") if cigar != "*" else len(seq)
            reads[("-" if flag & 16 else "+", pos, pos + span - 1, seq.upper())] += c
    if not reads and not dropped:
        raise SystemExit(f"\nNo aligned reads could be read from {path}. Expected columns: QNAME, FLAG, reference, position, CIGAR, sequence.")
    if not reads:                                                                         # e.g. a mock library whose only hits are shorter than min_read_len
        print(f"   WARNING: all {dropped:,} reads are shorter than {min_read_len} nt, so this sample has 0 usable reads (treated as zero counts)")
    print(f"   {lines:,} lines -> {sum(reads.values()):,} reads kept ({'read counts taken from the collapsed names' if collapsed else 'every line counted once'})"
          + (f"; {dropped:,} reads shorter than {min_read_len} nt ignored" if dropped else ""))
    return reads


def length_hist(reads, strand, region):
    h = collections.Counter()
    for (st, s, e, seq), c in reads.items():
        if st == strand and (region is None or (s >= region[0] and e <= region[1])):
            h[len(seq)] += c
    return h


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def add_stats(df, roles):
    """df has <role>_rep<r>_count columns; adds CPM per replicate, mean, SD and counts of replicates detected."""
    for role, (reps, totals) in roles.items():
        cc = [f"{role}_rep{r}_count" for r in reps]
        for c, r, t in zip(cc, reps, totals):
            df[f"{role}_rep{r}_CPM"] = df[c] / t * 1e6
        pc = [f"{role}_rep{r}_CPM" for r in reps]
        df[f"{role}_mean_CPM"] = df[pc].mean(axis=1)
        df[f"{role}_SD_CPM"] = df[pc].std(axis=1, ddof=1) if len(pc) > 1 else 0.0
        df[f"{role}_total_reads"] = df[cc].sum(axis=1)
        df[f"{role}_replicates_detected"] = (df[cc] > 0).sum(axis=1)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples"); ap.add_argument("--folder", default="aligned_tables")
    ap.add_argument("--group", required=True); ap.add_argument("--out", default="srna_results")
    ap.add_argument("--region", nargs=2, type=int)
    ap.add_argument("--min_len", type=int, default=15); ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--min_total", type=int, default=10); ap.add_argument("--mock_fraction", type=float, default=0.1)
    ap.add_argument("--min_read_len", type=int, default=18, help="ignore aligned reads shorter than this (default 18, the Cutadapt minimum)")
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    S = read_settings(a.samples)
    G = S[S["group"] == a.group]
    if G.empty:
        raise SystemExit(f"Group '{a.group}' not found. Available: {sorted(set(S['group']))}")
    data, roles = {}, {}
    for _, r in G.iterrows():
        path = find_file(a.folder, r["file"])
        print("reading", path)
        data[(r["role"], int(r["replicate"]))] = (load_sample(path, a.min_read_len), float(r["total_reads"]))
        roles.setdefault(r["role"], ([], []))
        roles[r["role"]][0].append(int(r["replicate"])); roles[r["role"]][1].append(float(r["total_reads"]))
    if "infected" not in roles:
        raise SystemExit("samples.csv has no 'infected' rows for this group.")

    # ---------- 1. read-length distribution ----------
    lens = np.arange(a.min_len, a.max_len + 1)
    scopes = [("all aligned reads", None)] + ([(f"3′ UTR ({a.region[0]}–{a.region[1]})", tuple(a.region))] if a.region else [])
    rows = []
    for sc_name, reg in scopes:
        for strand, sname in (("+", "sense"), ("-", "antisense")):
            d = pd.DataFrame({"scope": sc_name, "strand": sname, "length_nt": lens})
            for role, (reps, totals) in roles.items():
                for r in reps:
                    h = length_hist(data[(role, r)][0], strand, reg)
                    d[f"{role}_rep{r}_count"] = [h.get(L, 0) for L in lens]
                cc = [f"{role}_rep{r}_count" for r in reps]
                d[f"{role}_mean_count"] = d[cc].mean(axis=1)
                d[f"{role}_SD_count"] = d[cc].std(axis=1, ddof=1) if len(cc) > 1 else 0.0
                cpm = np.column_stack([d[c] / t * 1e6 for c, t in zip(cc, totals)])
                d[f"{role}_mean_CPM"] = cpm.mean(axis=1)
                d[f"{role}_SD_CPM"] = cpm.std(axis=1, ddof=1) if cpm.shape[1] > 1 else 0.0
            rows.append(d)
    LD = pd.concat(rows, ignore_index=True)
    LD.to_csv(os.path.join(a.out, f"{a.group}_length_distribution.csv"), index=False)

    unit = "count" if a.raw else "CPM"
    ylab = "reads (mean of replicates)" if a.raw else "reads per million (CPM)"
    fig, axes = plt.subplots(len(scopes), 2, figsize=(12, 3.6 * len(scopes)), squeeze=False)
    for i, (sc_name, _) in enumerate(scopes):
        for j, sname in enumerate(("sense", "antisense")):
            ax = axes[i, j]; style(ax)
            d = LD[(LD.scope == sc_name) & (LD.strand == sname)]
            ax.bar(d.length_nt, d[f"infected_mean_{unit}"], width=0.8, color=C_INF, label=f"infected (n={len(roles['infected'][0])}), mean ± SD", zorder=2)
            m_, sd_ = d[f"infected_mean_{unit}"].values, d[f"infected_SD_{unit}"].values
            ax.errorbar(d.length_nt, m_, yerr=np.vstack([np.minimum(sd_, m_), sd_]), fmt="none", ecolor=INK2, elinewidth=0.7, capsize=1.5, zorder=3)
            ax.set_ylim(bottom=0)
            if "mock" in roles:
                ax.plot(d.length_nt, d[f"mock_mean_{unit}"], color=C_MOCK, lw=1.2, marker="o", ms=3, label="mock", zorder=4)
            top = d.nlargest(2, f"infected_mean_{unit}")
            for _, t in top.iterrows():
                if t[f"infected_mean_{unit}"] > 0:
                    ax.text(t.length_nt, t[f"infected_mean_{unit}"] + t[f"infected_SD_{unit}"], f"{int(t.length_nt)}", ha="center", va="bottom", fontsize=8, color=INK)
            ax.set_xticks(range(a.min_len - a.min_len % 5 + 5 if a.min_len % 5 else a.min_len, a.max_len + 1, 5))
            ax.set_xlim(a.min_len - 1, a.max_len + 1)
            ax.set_title(f"{sc_name} – {sname}", loc="left", fontsize=10, color=INK)
            ax.set_ylabel(ylab, fontsize=8.5, color=INK2); ax.set_xlabel("read length (nt)", fontsize=8.5, color=INK2)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"{a.group}: length of aligned reads (each panel has its own y-scale)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(a.out, f"{a.group}_length_distribution.{ext}"), dpi=200)
    plt.close(fig)

    # ---------- 2. collapse into unique small RNAs ----------
    inf_total = collections.Counter()
    for r in roles["infected"][0]:
        inf_total.update(data[("infected", r)][0])
    keep = [k for k, c in inf_total.items() if c >= a.min_total]
    recs = []
    for k in keep:
        st, s, e, seq = k
        rec = {"strand": "sense" if st == "+" else "antisense", "start": s, "end": e, "length_nt": len(seq),
               "read_sequence_RNA": (seq if st == "+" else revcomp(seq)).replace("T", "U"),
               "genome_strand_sequence": seq.replace("T", "U")}
        for role, (reps, _) in roles.items():
            for r in reps:
                rec[f"{role}_rep{r}_count"] = data[(role, r)][0].get(k, 0)
        recs.append(rec)
    U = pd.DataFrame(recs)
    if U.empty:
        print("No unique small RNA reaches --min_total; lower it."); return
    U = add_stats(U, roles)
    mock_cpm = U["mock_mean_CPM"] if "mock" in roles else 0.0
    U["infection_specific"] = mock_cpm <= a.mock_fraction * U["infected_mean_CPM"]
    if a.region:
        U["in_region"] = (U.start >= a.region[0]) & (U.end <= a.region[1])
    U = U.sort_values("infected_mean_CPM", ascending=False).reset_index(drop=True)
    U.insert(0, "sRNA_id", [f"sRNA_{i + 1:04d}" for i in range(len(U))])
    U.to_csv(os.path.join(a.out, f"{a.group}_unique_srnas.csv"), index=False)

    # discrete loci: same strand, start and end (sequence variants with mismatches are added together)
    cnt_cols = [c for c in U.columns if c.endswith("_count")]
    L = U.groupby(["strand", "start", "end", "length_nt"], as_index=False)[cnt_cols].sum()
    L["n_sequence_variants"] = U.groupby(["strand", "start", "end", "length_nt"]).size().values
    L = add_stats(L, roles).sort_values("infected_mean_CPM", ascending=False).reset_index(drop=True)
    L.insert(0, "locus_id", [f"locus_{i + 1:04d}" for i in range(len(L))])
    L.to_csv(os.path.join(a.out, f"{a.group}_unique_srna_loci.csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(a.out, f"{a.group}_unique_srnas.xlsx")) as xw:
            U.head(5000).to_excel(xw, sheet_name="unique_srnas", index=False)
            L.head(5000).to_excel(xw, sheet_name="loci", index=False)
            LD.to_excel(xw, sheet_name="length_distribution", index=False)
        xl = " and .xlsx"
    except ModuleNotFoundError:
        xl = " (install openpyxl for .xlsx)"

    sense = LD[(LD.scope == scopes[0][0]) & (LD.strand == "sense")]
    mode = int(sense.loc[sense["infected_mean_CPM"].idxmax(), "length_nt"])
    print(f"\n{a.group}: most common sense read length (infected) = {mode} nt")
    print(f"{len(U)} unique small RNAs with >= {a.min_total} reads in infected samples; {int(U.infection_specific.sum())} are infection-specific.")
    print("\nTop 8 by CPM:")
    print(U.head(8)[["sRNA_id", "strand", "start", "end", "length_nt", "infected_mean_CPM", "infected_replicates_detected", "infection_specific"]].round(1).to_string(index=False))
    print(f"\nSaved to {a.out}/ as .csv, .png, .svg{xl}")


if __name__ == "__main__":
    main()
