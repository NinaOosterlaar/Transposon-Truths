#!/usr/bin/env python3
import argparse, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless for Slurm jobs
import matplotlib.pyplot as plt
import seaborn as sns

chromosome_mapper = {
    "NC_001133": "chrI",   "NC_001134": "chrII",  "NC_001135": "chrIII",
    "NC_001136": "chrIV",  "NC_001137": "chrV",   "NC_001138": "chrVI",
    "NC_001139": "chrVII", "NC_001140": "chrVIII","NC_001141": "chrIX",
    "NC_001142": "chrX",   "NC_001143": "chrXI",  "NC_001144": "chrXII",
    "NC_001145": "chrXIII","NC_001146": "chrXIV", "NC_001147": "chrXV",
    "NC_001148": "chrXVI", "NC_001224": "chrM",
}

yeast_chrom_lengths = {
    "chrI": 230218,  "chrII": 813184,  "chrIII": 316620, "chrIV": 1531933,
    "chrV": 576874,  "chrVI": 270161,  "chrVII": 1090940,"chrVIII": 562643,
    "chrIX": 439888, "chrX": 745751,   "chrXI": 666816,  "chrXII": 1078177,
    "chrXIII": 924431,"chrXIV": 784333,"chrXV": 1091291, "chrXVI": 948066,
    "chrM": 85779,
}

def ensure_dir(path: str):
    path = os.path.expanduser(path)  # expand '~' if present
    os.makedirs(path, exist_ok=True)
    return path

def read_wig(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    chrom_data, current_data = {}, []
    current_chrom = None
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('track'):
                continue
            if line.startswith(('VariableStep', 'variableStep', 'fixedStep')):
                if current_chrom and current_data:
                    chrom_data[current_chrom] = pd.DataFrame(current_data, columns=['Position','Value'])
                    current_data = []
                for ncbi_id, mapped in chromosome_mapper.items():
                    if ncbi_id in line or mapped in line:
                        current_chrom = mapped
                        break
                continue
            parts = line.strip().split()
            if len(parts) == 2 and current_chrom:
                pos, val = parts
                current_data.append((int(pos), float(val)))
    if current_chrom and current_data:
        chrom_data[current_chrom] = pd.DataFrame(current_data, columns=['Position','Value'])
    return chrom_data

def label_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    cuts = [p for p in (stem.find('_merged'), stem.find('_FDDP'),
                        stem.find('-merged'), stem.find('-FDDP')) if p != -1]
    head = stem[:min(cuts)] if cuts else stem
    m = re.search(r'(?i)\bFD(\d+)[-_](\d+)\b', head)
    if m: return f"FD{m.group(1)}_{m.group(2)}"
    m = re.search(r'(?i)\byLIC(\d+)[-_](\d+)\b', head)
    if m: return f"yLIC{m.group(1)}_{m.group(2)}"
    m_strain = re.search(r'(?i)\bdnrp(\d+)\b', head)
    if m_strain:
        strain = m_strain.group(1)
        m_rep = re.search(r'[._-](\d+)$', head)
        rep = m_rep.group(1) if m_rep else None
        m_letter = re.search(r'(?i)(?:^|[-_])(a|b)(?:$|[-_])', stem)
        letter = m_letter.group(1).lower() if m_letter else None
        if rep and letter: return f"dnrp{strain}-{rep}-{letter}"
        if letter: return f"dnrp{strain}-{letter}"
        if rep: return f"dnrp{strain}_{rep}"
        return f"dnrp{strain}"
    return head

def plot_basic_statistics(folder, outdir):
    wig_files = sorted([f for f in os.listdir(folder) if f.endswith('.wig')])
    stats = []
    genome_size = sum(yeast_chrom_lengths.values())
    for wig in wig_files:
        d = read_wig(os.path.join(folder, wig))
        total = 0
        occupied = 0
        for _, df in d.items():
            total += df['Value'].sum()
            if not df.empty:
                occupied += (df['Value'] > 0).sum()
        mean_count = total / genome_size
        unocc = genome_size - occupied
        stats.append((label_from_filename(wig), total, mean_count, occupied, unocc))
    if not stats:
        print("No WIG files found for basic statistics."); return
    labels  = [s[0] for s in stats]
    totals  = [s[1] for s in stats]
    means   = [s[2] for s in stats]
    occ     = [s[3] for s in stats]
    unocc   = [s[4] for s in stats]

    plt.figure(figsize=(10,6))
    plt.bar(labels, totals)
    plt.title('Total Sum of Counts per WIG File')
    plt.xlabel('Sample'); plt.ylabel('Total Counts')
    plt.xticks(rotation=45, ha="right"); plt.grid(axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "total_sum_per_file.png"), dpi=200); plt.close()

    plt.figure(figsize=(10,6))
    plt.bar(labels, means)
    plt.title('Mean Coverage per Base Pair per WIG File')
    plt.xlabel('Sample'); plt.ylabel('Mean Coverage per bp')
    plt.xticks(rotation=45, ha="right"); plt.grid(axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mean_coverage_per_bp.png"), dpi=200); plt.close()

    plt.figure(figsize=(12,6))
    plt.bar(labels, occ, label="Occupied sites")
    plt.bar(labels, unocc, bottom=occ, label="Unoccupied sites")
    plt.title('Occupied vs Unoccupied Sites per Sample')
    plt.xlabel('Sample'); plt.ylabel('Number of Sites')
    plt.xticks(rotation=45, ha="right"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "occupied_vs_unoccupied.png"), dpi=200); plt.close()

def show_counts_part_chromosome(file, chrom, start, end, outdir):
    d = read_wig(file)
    if chrom not in d:
        print(f"Chromosome {chrom} not found in data."); return
    df = d[chrom]
    region = df[(df['Position'] >= start) & (df['Position'] <= end)]
    plt.figure(figsize=(12,6))
    plt.bar(region['Position'], region['Value'], width=1)
    plt.title(f'Counts for {chrom}:{start}-{end}')
    plt.xlabel('Position'); plt.ylabel('Counts')
    plt.xlim(start, end); plt.grid(axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"counts_{chrom}_{start}-{end}.png"), dpi=200)
    plt.close()

def plot_presence_transposon(folder, chrom, start, end, outdir):
    all_counts = []
    for file in sorted(os.listdir(folder)):
        if not file.endswith('.wig'): continue
        d = read_wig(os.path.join(folder, file))
        if chrom not in d:
            print(f"Chromosome {chrom} not found in {file}. Skipping."); continue
        df = d[chrom]
        region = df[(df['Position'] >= start) & (df['Position'] <= end)]
        presence = np.zeros(end - start + 1, dtype=int)
        for _, row in region.iterrows():
            pos = int(row['Position']) - start
            if 0 <= pos < len(presence): presence[pos] = 1
        all_counts.append((label_from_filename(file), presence))
    if not all_counts:
        print("No valid data found for the specified chromosome and region."); return
    all_counts.sort(key=lambda x: x[0])
    labels = [x[0] for x in all_counts]
    data = np.array([x[1] for x in all_counts])
    plt.figure(figsize=(12, max(6, len(labels) * 0.3)))
    sns.heatmap(data, cmap='Greys', cbar=False, yticklabels=labels)
    plt.title(f'Transposon Insertion Presence in {chrom}:{start}-{end}')
    plt.xlabel('Position in Region'); plt.ylabel('Samples')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"presence_{chrom}_{start}-{end}.png"), dpi=200)
    plt.close()

def main():
    p = argparse.ArgumentParser(description="Generate SATAY figures and save to a figures directory.")
    p.add_argument("--data-dir", required=True, help="Folder containing .wig files")
    p.add_argument("--chrom", default="chrI")
    p.add_argument("--start", type=int, default=10000)
    p.add_argument("--end", type=int, default=20000)
    p.add_argument("--also-counts", action="store_true",
                   help="Also export a bar plot of counts for the region from the first WIG file.")
    p.add_argument("--fig-dir", default=None,
                   help="Optional output folder for figures. If omitted, uses <data-dir>/figures")
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data dir not found: {data_dir}")

    # If --fig-dir given, use it; else fall back to <data-dir>/figures
    figures_dir = ensure_dir(os.path.expanduser(args.fig_dir) if args.fig_dir
                             else os.path.join(data_dir, "figures"))

    # 1) Summary stats across all .wig files
    plot_basic_statistics(data_dir, figures_dir)

    # 2) Heatmap presence/absence in region
    plot_presence_transposon(data_dir, args.chrom, args.start, args.end, figures_dir)

    # 3) Optional counts plot for first WIG file
    if args.also_counts:
        wigs = sorted([f for f in os.listdir(data_dir) if f.endswith(".wig")])
        if wigs:
            show_counts_part_chromosome(
                os.path.join(data_dir, wigs[0]), args.chrom, args.start, args.end, figures_dir
            )
        else:
            print("No WIG files found for --also-counts.")

    print(f"All figures written to: {figures_dir}")

if __name__ == "__main__":
    main()