#!/usr/bin/env python3
"""
Peak table + fragment pairing for one group (e.g. HEK_KUNV).

Uses the results made by normalise_and_plot.py:
    results/<group>_5prime_CPM.csv   (where fragments START)
    results/<group>_3prime_CPM.csv   (where fragments END)
plus the virus FASTA (to read the nucleotide at each peak) and samples.csv (to convert CPM back to raw read counts).

What it does
  1. finds peaks in the 5' and 3' end tracks (local maxima above a CPM threshold)
  2. for each peak: position, nucleotide, mean CPM, SD, CPM of each replicate, how many replicates support it,
     approximate raw read count, mock CPM, and a PASS flag (reproducible, enough reads, absent in mock)
  3. pairs every 5' peak with every 3' peak downstream of it within a length window -> fragment length + sequence

Usage (run from your project folder):
    python find_peaks_and_fragments.py --group HEK_KUNV --fasta KUNV.fasta --samples samples.csv --region 10391 11015

Important: positions are 1-based and the FASTA must be the same sequence you mapped to (same length as the tables).
Peaks are only meaningful if the 5'/3' tables were made for the SENSE strand (genomecov strand = +).
"""
import argparse, os, re
import numpy as np
import pandas as pd


def read_settings(path):
    """Read samples.csv saved by VS Code or Excel (handles , or ; and a BOM)."""
    first = open(path, encoding="utf-8-sig").readline()
    sep = ";" if first.count(";") > first.count(",") else ","
    return pd.read_csv(path, sep=sep, encoding="utf-8-sig")


def read_fasta(path):
    name, seq = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                break                      # only the first record
            name = line[1:]
        elif line:
            seq.append(line.upper().replace("T", "U"))   # show as RNA
    return name, "".join(seq)


def call_peaks(df, tag, seq, totals, a):
    rep_cols = [c for c in df.columns if re.fullmatch(r"infected_rep\d+_CPM", c)]
    mean = df["infected_mean_CPM"]
    mock = df["mock_mean_CPM"] if "mock_mean_CPM" in df.columns else pd.Series(0.0, index=df.index)
    # a position is a peak if it is the maximum within +/- window and above the CPM threshold
    local_max = mean.rolling(2 * a.window + 1, center=True, min_periods=1).max()
    cand = df[(mean >= a.min_cpm) & (mean == local_max)]
    if a.region:
        cand = cand[(cand.position >= a.region[0]) & (cand.position <= a.region[1])]
    cand = cand.sort_values("infected_mean_CPM", ascending=False)
    kept = []                                          # greedy: strongest first, drop neighbours within window
    for _, r in cand.iterrows():
        if all(abs(r.position - p) > a.window for p in kept):
            kept.append(int(r.position))
        if len(kept) >= a.top:
            break
    rows = []
    for p in kept:
        r = df[df.position == p].iloc[0]
        rep_vals = [float(r[c]) for c in rep_cols]
        n_support = sum(v >= a.min_rep_cpm for v in rep_vals)
        raw = [v * t / 1e6 for v, t in zip(rep_vals, totals)] if totals else None
        total_raw = sum(raw) if raw else np.nan
        nuc = seq[p - 1] if p <= len(seq) else "?"
        if tag == "5prime":      # cut happens between the base before the peak and the peak base
            up = seq[p - 2] if p >= 2 else "-"
            context = f"{up}|{nuc}"
        else:                    # 3' end: cut happens between the peak base and the next base
            down = seq[p] if p < len(seq) else "-"
            context = f"{nuc}|{down}"
        m = float(mock.loc[r.name])
        ok = (n_support >= a.min_reps) and (m <= a.mock_fraction * float(r["infected_mean_CPM"])) \
             and (np.isnan(total_raw) or total_raw >= a.min_reads)
        row = {"end_type": tag, "position": p, "nucleotide": nuc, "cut_site_context": context,
               "mean_CPM": round(float(r["infected_mean_CPM"]), 2), "SD_CPM": round(float(r["infected_SD"]), 2)}
        for i, v in enumerate(rep_vals, 1):
            row[f"rep{i}_CPM"] = round(v, 2)
        if raw:
            for i, v in enumerate(raw, 1):
                row[f"rep{i}_approx_reads"] = round(v)
        row.update({"replicates_supporting": n_support, "approx_total_reads": None if np.isnan(total_raw) else round(total_raw),
                    "mock_mean_CPM": round(m, 3), "PASS": ok})
        rows.append(row)
    return pd.DataFrame(rows)


def pair_fragments(p5, p3, seq, a):
    out = []
    p5 = p5[p5.PASS] if a.pass_only else p5
    p3 = p3[p3.PASS] if a.pass_only else p3
    for _, r5 in p5.iterrows():
        partners = p3[(p3.position >= r5.position + a.min_len - 1) & (p3.position <= r5.position + a.max_len - 1)]
        for _, r3 in partners.iterrows():
            L = int(r3.position - r5.position + 1)
            out.append({"start_5prime_pos": int(r5.position), "start_CPM": r5.mean_CPM, "start_context": r5.cut_site_context,
                        "end_3prime_pos": int(r3.position), "end_CPM": r3.mean_CPM, "end_context": r3.cut_site_context,
                        "fragment_length_nt": L, "pair_score_min_CPM": min(r5.mean_CPM, r3.mean_CPM),
                        "fragment_sequence": seq[int(r5.position) - 1:int(r3.position)]})
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("pair_score_min_CPM", ascending=False)
        df["partner_rank_for_this_start"] = df.groupby("start_5prime_pos")["end_CPM"].rank(ascending=False, method="first").astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, help="group name exactly as in samples.csv, e.g. HEK_KUNV")
    ap.add_argument("--fasta", required=True, help="FASTA of the virus this group was mapped to")
    ap.add_argument("--samples", help="samples.csv (gives total reads so CPM can be converted to raw reads)")
    ap.add_argument("--results", default="results", help="folder with the *_CPM.csv files (default: results)")
    ap.add_argument("--out", default="peaks", help="output folder (default: peaks)")
    ap.add_argument("--region", nargs=2, type=int, help="only analyse this window, e.g. 10391 11015 (3' UTR)")
    ap.add_argument("--min_cpm", type=float, default=5, help="minimum mean CPM for a peak (default 5)")
    ap.add_argument("--min_rep_cpm", type=float, default=2, help="CPM a replicate needs to 'support' a peak (default 2)")
    ap.add_argument("--min_reps", type=int, default=2, help="replicates that must support a peak (default 2 of 3)")
    ap.add_argument("--min_reads", type=float, default=10, help="minimum approx. total reads at a peak (default 10)")
    ap.add_argument("--mock_fraction", type=float, default=0.1, help="mock CPM must be below this fraction of infected (default 0.1)")
    ap.add_argument("--window", type=int, default=3, help="peaks are the maximum within +/- this many nt (default 3)")
    ap.add_argument("--top", type=int, default=60, help="keep at most this many peaks per end (default 60)")
    ap.add_argument("--min_len", type=int, default=14, help="shortest fragment to pair (default 14)")
    ap.add_argument("--max_len", type=int, default=120, help="longest fragment to pair (default 120)")
    ap.add_argument("--pass_only", action="store_true", help="pair only peaks that passed the filters")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fa_name, seq = read_fasta(a.fasta)
    tabs = {}
    for tag in ("5prime", "3prime"):
        path = os.path.join(a.results, f"{a.group}_{tag}_CPM.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Cannot find {path}. Run normalise_and_plot.py with --tag {tag} first.")
        tabs[tag] = pd.read_csv(path)
        if len(tabs[tag]) != len(seq):
            raise SystemExit(f"\nERROR: {path} has {len(tabs[tag])} positions but the FASTA ({fa_name}) has {len(seq)} nt.\n"
                             "The FASTA must be the exact sequence that was mapped to (same length). Check the file and virus.")

    totals = None
    if a.samples:
        S = read_settings(a.samples)
        S = S[(S["group"] == a.group) & (S["role"] == "infected")].sort_values("replicate")
        totals = S["total_reads"].astype(float).tolist()

    p5 = call_peaks(tabs["5prime"], "5prime", seq, totals, a)
    p3 = call_peaks(tabs["3prime"], "3prime", seq, totals, a)
    pairs = pair_fragments(p5, p3, seq, a)

    p5.to_csv(os.path.join(a.out, f"{a.group}_5prime_peaks.csv"), index=False)
    p3.to_csv(os.path.join(a.out, f"{a.group}_3prime_peaks.csv"), index=False)
    pairs.to_csv(os.path.join(a.out, f"{a.group}_fragment_pairs.csv"), index=False)
    try:                                               # one Excel file with 3 sheets, if openpyxl is installed
        with pd.ExcelWriter(os.path.join(a.out, f"{a.group}_peaks.xlsx")) as xw:
            p5.to_excel(xw, sheet_name="5prime_peaks", index=False)
            p3.to_excel(xw, sheet_name="3prime_peaks", index=False)
            pairs.to_excel(xw, sheet_name="fragment_pairs", index=False)
        xl = " and .xlsx"
    except ModuleNotFoundError:
        xl = " (install openpyxl for an .xlsx version)"

    print(f"\n{a.group}: {len(p5)} 5' peaks ({int(p5.PASS.sum()) if len(p5) else 0} pass), "
          f"{len(p3)} 3' peaks ({int(p3.PASS.sum()) if len(p3) else 0} pass), {len(pairs)} candidate fragments.")
    if not pairs.empty:
        print("\nTop 5 candidate fragments:")
        print(pairs.head(5)[["start_5prime_pos", "end_3prime_pos", "fragment_length_nt", "start_CPM", "end_CPM"]].to_string(index=False))
    print(f"\nSaved to {a.out}/ as .csv{xl}")


if __name__ == "__main__":
    main()
