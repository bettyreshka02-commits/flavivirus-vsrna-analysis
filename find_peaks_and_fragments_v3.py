#!/usr/bin/env python3
"""
find_peaks_and_fragments_v3.py: the same peak calling and fragment pairing as find_peaks_and_fragments.py,
plus an EXACT-READ SUPPORT check for every candidate fragment.

Why: pairing a 5' peak with a 3' peak only PROPOSES a fragment. It does not show that any single RNA starts at
the first peak AND ends at the second. A pair is kept as a fragment only if reads exist that start exactly at the
5' peak and end exactly at the 3' peak.

New columns in <group>_fragment_pairs.csv
    exact_reads_CPM    mean CPM (over the infected replicates) of sense reads that start at start_5prime_pos and end at end_3prime_pos
    near_reads_CPM     the same, allowing the start and the end to differ by up to --near nt (default 2)
    supported          True if near_reads_CPM >= --min_exact (default 5), i.e. reads whose start and end are each within +-NEAR nt (default 2) of the peaks
    supported_exact    True if exact_reads_CPM >= --min_exact (the strict v2 rule, kept for comparison)
v3 change: the support rule now tolerates a +-2 nt shift at each end (supervisor advice) instead of requiring exact ends.
The pair list is sorted by near_reads_CPM, and <group>_fragments_supported.csv holds only the supported pairs.

Usage (run from the project folder; the aligned 6-column tables must be in --aligned):
    python find_peaks_and_fragments_v3.py --group HEK_KUNV --fasta "reference files/WNV_KUN_NSW2011.fa" --samples samples.csv --pass_only --region 10399 11015 --out peaks_v3
Whole genome: leave out --region and use --top 5000 (the default of 60 peaks per end cuts off most coding-region peaks).
For A549 use the tables realigned to NSW2011 (--aligned update_2026-10-26/aligned_tables_A549_realigned); the old FLSDX
tables have shifted numbering above about 10,455.
Note: the +-NEAR windows of neighbouring pairs overlap, so near_reads_CPM is not additive across pairs.
"""
import argparse, os
import numpy as np
import pandas as pd
import find_peaks_and_fragments as base   # the original script must be in the same folder


def load_reads(aligned_dir, samples, group):
    """Return a list (one per infected replicate) of dicts {(start, end): CPM} for sense, ungapped alignments."""
    S = samples[(samples["group"] == group) & (samples["role"] == "infected")].sort_values("replicate")
    out = []
    for r in S.itertuples():
        d = pd.read_csv(os.path.join(aligned_dir, r.file), sep="\t", header=None,
                        names=["name", "flag", "ref", "pos", "cigar", "seq"])
        d = d[(d["flag"] == 0) & d["cigar"].astype(str).str.fullmatch(r"\d+M")].copy()
        d["count"] = d["name"].astype(str).str.rsplit("-", n=1).str[1].astype(int)   # collapsed reads are named <id>-<count>
        d["end"] = d["pos"] + d["seq"].str.len() - 1
        cpm = d.groupby(["pos", "end"])["count"].sum() / float(r.total_reads) * 1e6
        out.append(cpm.to_dict())
    return out


def exact_support(pairs, reads, near):
    ex, nr = [], []
    for r in pairs.itertuples():
        a, b = int(r.start_5prime_pos), int(r.end_3prime_pos)
        ex.append(np.mean([rep.get((a, b), 0.0) for rep in reads]))
        nr.append(np.mean([sum(rep.get((a + i, b + j), 0.0) for i in range(-near, near + 1) for j in range(-near, near + 1))
                           for rep in reads]))
    pairs = pairs.copy()
    pairs["exact_reads_CPM"] = np.round(ex, 2)
    pairs["near_reads_CPM"] = np.round(nr, 2)
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--aligned", default="aligned_tables", help="folder with the aligned 6-column tables named in samples.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="peaks")
    ap.add_argument("--region", nargs=2, type=int)
    ap.add_argument("--min_cpm", type=float, default=5)
    ap.add_argument("--min_rep_cpm", type=float, default=2)
    ap.add_argument("--min_reps", type=int, default=2)
    ap.add_argument("--min_reads", type=float, default=10)
    ap.add_argument("--mock_fraction", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--min_len", type=int, default=14)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--pass_only", action="store_true")
    ap.add_argument("--min_exact", type=float, default=5, help="CPM of reads (start and end within +-near nt) a pair needs to count as a fragment (default 5)")
    ap.add_argument("--near", type=int, default=2, help="tolerance in nt at each end for near_reads_CPM and for support (default 2)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    _, seq = base.read_fasta(a.fasta)
    tabs = {}
    for tag in ("5prime", "3prime"):
        path = os.path.join(a.results, f"{a.group}_{tag}_CPM.csv")
        tabs[tag] = pd.read_csv(path)
        if len(tabs[tag]) != len(seq):
            raise SystemExit(f"ERROR: {path} has {len(tabs[tag])} positions but the FASTA has {len(seq)} nt.")
    samples = base.read_settings(a.samples)
    S = samples[(samples["group"] == a.group) & (samples["role"] == "infected")].sort_values("replicate")
    totals = S["total_reads"].astype(float).tolist()

    p5 = base.call_peaks(tabs["5prime"], "5prime", seq, totals, a)
    p3 = base.call_peaks(tabs["3prime"], "3prime", seq, totals, a)
    pairs = base.pair_fragments(p5, p3, seq, a)
    if pairs.empty:
        raise SystemExit("No candidate pairs.")
    reads = load_reads(a.aligned, samples, a.group)
    pairs = exact_support(pairs, reads, a.near)
    pairs["supported_exact"] = pairs["exact_reads_CPM"] >= a.min_exact
    pairs["supported"] = pairs["near_reads_CPM"] >= a.min_exact
    pairs = pairs.sort_values(["near_reads_CPM", "exact_reads_CPM", "pair_score_min_CPM"], ascending=False)

    p5.to_csv(os.path.join(a.out, f"{a.group}_5prime_peaks.csv"), index=False)
    p3.to_csv(os.path.join(a.out, f"{a.group}_3prime_peaks.csv"), index=False)
    pairs.to_csv(os.path.join(a.out, f"{a.group}_fragment_pairs.csv"), index=False)
    pairs[pairs["supported"]].to_csv(os.path.join(a.out, f"{a.group}_fragments_supported.csv"), index=False)
    print(f"\n{a.group}: {len(pairs)} candidate pairs, {int(pairs['supported'].sum())} with at least {a.min_exact} CPM of reads within +-{a.near} nt ({int(pairs['supported_exact'].sum())} with exact reads).")
    cols = ["start_5prime_pos", "end_3prime_pos", "fragment_length_nt", "start_CPM", "end_CPM", "exact_reads_CPM", "near_reads_CPM", "supported", "supported_exact"]
    print(pairs[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
