import sgd
import json
import os
from collections import defaultdict
import numpy as np

representative_genes = [
    "PRM9",  # Chr I
    "ALG3",   # Chr II
    "CDC10",     # Chr III
    "HO",   # Chr IV
    "RNR1",   # Chr V
    "SMC1",   # Chr VI
    "CUP2",   # Chr VII
    "OPI1",   # Chr VIII
    "ATG32",   # Chr IX
    "TDH2",   # Chr X
    "MLP1",   # Chr XI
    "RDN25-1",# Chr XII
    "PHO84",  # Chr XIII
    "MRPL50",  # Chr XIV
    "HIS3",   # Chr XV
    "SSN3"   # Chr XVI
]

mapping_to_roman = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    13: "XIII",
    14: "XIV",
    15: "XV",
    16: "XVI"
}

chromosome_length = {
    "ChrI": 230218,
    "ChrII": 813184,
    "ChrIII": 316620,
    "ChrIV": 1531933,
    "ChrV": 576874,
    "ChrVI": 270161,
    "ChrVII": 1090940,
    "ChrVIII": 562643,
    "ChrIX": 439888,
    "ChrX": 745751,
    "ChrXI": 666816,
    "ChrXII": 1078171,
    "ChrXIII": 924431,
    "ChrXIV": 784333,
    "ChrXV": 1091291,
    "ChrXVI": 948066,
    "ChrM": 85779,          # mitochondrial genome (approx for S288C)
    "2micron": 6318         # 2-micron plasmid
}

        
class Centromeres:
    def __init__(self, centromere_file):
        with open(centromere_file, 'r') as f:
            self.centromeres = json.load(f)
    
    def get_centromere(self, chromosome):
        """Return the centromere information for a given chromosome."""
        return self.centromeres.get(chromosome, None)
    
    def list_all_centromeres(self):
        """Return the full dictionary of centromeres."""
        return self.centromeres
    
    def retrieve_middles(self):
        """Return a dictionary of chromosome names and their centromere middles."""
        return {chrom: info["middle"] for chrom, info in self.centromeres.items()}
    
    def retrieve_lengths(self):
        """Return a dictionary of chromosome names and their centromere lengths."""
        return {chrom: info["length"] for chrom, info in self.centromeres.items()}
    
    def retrieve_starts(self):
        """Return a dictionary of chromosome names and their centromere starts."""
        return {chrom: info["start"] for chrom, info in self.centromeres.items()}
    
    def retrieve_ends(self):
        """Return a dictionary of chromosome names and their centromere ends."""
        return {chrom: info["end"] for chrom, info in self.centromeres.items()}
    
    def retrieve_all_middles(self):
        """Return a list of all centromere middles."""
        middles = {}
        for chrom in self.centromeres:
            middles[chrom] = self.centromeres[chrom]["middle"]
        return middles

    def retrieve_all_lengths(self):
        """Return a list of all centromere lengths."""
        lengths = {}
        for chrom in self.centromeres:
            lengths[chrom] = self.centromeres[chrom]["length"]
        return lengths
    
    def retrieve_all_starts(self):
        """Return a list of all centromere starts."""
        starts = {}
        for chrom in self.centromeres:
            starts[chrom] = self.centromeres[chrom]["start"]
        return starts
    
    def retrieve_all_ends(self):
        """Return a list of all centromere ends."""
        ends = {}
        for chrom in self.centromeres:
            ends[chrom] = self.centromeres[chrom]["end"]
        return ends
    
    
class Nucleosomes:
    def __init__(self, nucleosome_dir):
        """Load all chromosome nucleosome files from a directory."""
        self.nucleosomes = {}
        for chrom in chromosome_length.keys():
            file_path = os.path.join(nucleosome_dir, f"nucleosome_data_{chrom}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.nucleosomes[chrom] = json.load(f)
    
    def get_nucleosomes(self, chrom):
        """Return the list of nucleosomes for a given chromosome."""
        return self.nucleosomes.get(chrom, [])
    
    def list_all_nucleosomes(self):
        """Return all nucleosomes for all chromosomes."""
        return self.nucleosomes
    
    def get_all_middles(self):
        """Return dict of chromosome names → list of nucleosome middles."""
        return {chrom: [nuc[1] for nuc in nucs] for chrom, nucs in self.nucleosomes.items()}
    
    def count_nucleosomes(self, chrom):
        """Return the number of nucleosomes on a given chromosome."""
        return len(self.nucleosomes.get(chrom, []))
    
    def get_middles(self, chrom):
        """Return a list of nucleosome middles on a given chromosome."""
        return [nuc[1] for nuc in self.nucleosomes.get(chrom, [])]
    
    def compute_average_span(self, chrom):
        """Compute the average span of nucleosomes on a given chromosome."""
        number_of_nucleosomes = self.count_nucleosomes(chrom)
        return chromosome_length[chrom] / number_of_nucleosomes if number_of_nucleosomes > 0 else 0
    
    
def create_centromere_dict(output_file):
    """Create a dictionary of centromere locations for each chromosome."""
    centromeres = {}
    for i, gene in enumerate(representative_genes, start=1):
        chrom_name = "Chromosome_" + mapping_to_roman[i]
        info = sgd.gene(gene).sequence_details.json()
        start = info['genomic_dna'][0]["contig"]['centromere_start']
        end = info['genomic_dna'][0]["contig"]['centromere_end']
        middle = (start + end) // 2
        length = end - start
        centromeres[chrom_name] = {
            "start": start,
            "end": end,
            "middle": middle,
            "length": length
        }
    with open(output_file, 'w') as f:
        json.dump(centromeres, f, indent=4)
        

