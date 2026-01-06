import pandas as pd


class NucleosomePositionReader:
    def __init__(self, path):
        self.nucleosome_positions = {
            'chromosome ID': [],
            'position': [],
            'NCP score': [],
            'NCP score/noise ratio': [],
        }
        self.path = path
        self.read_table()
        self.df = self.get_positions_as_dataframe()

    def read_table(self):
        with open(self.path) as file:
            for line in file.readlines():
                line = line.strip('\n')
                chromosome_ID, position, NCP_score, NCP_score_to_noise = line.split()
                self.nucleosome_positions['chromosome ID'].append(chromosome_ID)
                self.nucleosome_positions['position'].append(int(position))
                self.nucleosome_positions['NCP score'].append(float(NCP_score))
                self.nucleosome_positions['NCP score/noise ratio'].append(float(NCP_score_to_noise))

    def get_positions_as_dataframe(self):
        return pd.DataFrame(self.nucleosome_positions)
