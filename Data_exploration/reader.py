import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns  

chromosome_mapper = {
    "NC_001133": "chrI",
    "NC_001134": "chrII",
    "NC_001135": "chrIII",
    "NC_001136": "chrIV",
    "NC_001137": "chrV",
    "NC_001138": "chrVI",
    "NC_001139": "chrVII",
    "NC_001140": "chrVIII",
    "NC_001141": "chrIX",
    "NC_001142": "chrX",
    "NC_001143": "chrXI",
    "NC_001144": "chrXII",
    "NC_001145": "chrXIII",
    "NC_001146": "chrXIV",
    "NC_001147": "chrXV",
    "NC_001148": "chrXVI",
    "NC_001224": "chrM",
}

def read_wig(file_path):
    """
    Reads a WIG file and returns a dict of DataFrames, one per chromosome.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    chrom_data = {}
    current_chrom = None
    current_data = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('track'):
                continue  # skip metadata lines

            if line.startswith(('VariableStep', 'variableStep', 'fixedStep')):
                # Save previous chromosome before moving on
                if current_chrom and current_data:
                    chrom_data[current_chrom] = pd.DataFrame(
                        current_data, columns=['Position', 'Value']
                    )
                    current_data = []

                # Detect chromosome ID in line
                for ncbi_id, mapped_chrom in chromosome_mapper.items():
                    if ncbi_id in line or mapped_chrom in line:
                        current_chrom = mapped_chrom
                        break
                continue  # skip definition line

            # Parse data lines
            parts = line.strip().split()
            if len(parts) == 2 and current_chrom:
                position, value = parts
                current_data.append((int(position), float(value)))

    # Save last chromosome
    if current_chrom and current_data:
        chrom_data[current_chrom] = pd.DataFrame(
            current_data, columns=['Position', 'Value']
        )

    return chrom_data


def label_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]

    # cut before merged/FDDP tokens
    cut_positions = [p for p in (
        stem.find('_merged'), stem.find('_FDDP'),
        stem.find('-merged'), stem.find('-FDDP')
    ) if p != -1]
    head = stem[:min(cut_positions)] if cut_positions else stem

    # FD{strain}_{rep}
    m = re.search(r'(?i)\bFD(\d+)[-_](\d+)\b', head)
    if m:
        return f"FD{m.group(1)}_{m.group(2)}"

    # yLIC{strain}_{rep}
    m = re.search(r'(?i)\byLIC(\d+)[-_](\d+)\b', head)
    if m:
        return f"yLIC{m.group(1)}_{m.group(2)}"

    # dnrp{strain}-{repnum} ... {a|b}
    m_strain = re.search(r'(?i)\bdnrp(\d+)\b', head)
    if m_strain:
        strain = m_strain.group(1)
        # numeric replicate before _merged
        m_repnum = re.search(r'[._-](\d+)$', head)
        repnum = m_repnum.group(1) if m_repnum else None
        # a/b letter anywhere in full stem
        m_letter = re.search(r'(?i)(?:^|[-_])(a|b)(?:$|[-_])', stem)
        letter = m_letter.group(1).lower() if m_letter else None

        if repnum and letter:
            return f"dnrp{strain}-{repnum}-{letter}"  # UNIQUE
        if letter:
            return f"dnrp{strain}-{letter}"
        if repnum:
            return f"dnrp{strain}_{repnum}"
        return f"dnrp{strain}"

    # fallback: keep head
    return head
        

if __name__ == "__main__":
# Example of showing counts for a specific region
    # show_counts_part_chromosome("Data/E-MTAB-14476/FD7_1_FDDP210435821-2a_HTWL7DSX2_L4_trimmed_forward_notrimmed.sorted.bam.wig", "chrI", 10000, 20000)  # Change as needed
    pass 