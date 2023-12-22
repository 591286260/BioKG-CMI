import pandas as pd
from tqdm import tqdm
import random

def read_fasta(fasta_file):

    data = {}
    current_id = None
    with open(fasta_file, 'r') as file:
        for line in file:
            if line.startswith('>'):
                current_id = line.strip()[1:]
            else:
                data[current_id] = line.strip()
    return data

def read_xlsx(xlsx_file):

    data = {}
    df = pd.read_excel(xlsx_file, header=None)
    for _, row in df.iterrows():
        miRNA_id = row[0]
        subcellular_location = row[1]
        data[miRNA_id] = data.get(miRNA_id, []) + [subcellular_location]

def read_positive_samples(csv_file):
    df = pd.read_csv(csv_file, header=None)
def generate_negative_samples(circRNA_data, miRNA_data, positive_samples):

    all_samples = []
    circRNA_ids = list(circRNA_data.keys())
    miRNA_ids = list(miRNA_data.keys())

    for circRNA_id in circRNA_ids:
        for miRNA_id in miRNA_ids:
            all_samples.append((circRNA_id, miRNA_id))

    all_samples_set = set(all_samples)
    positive_samples_set = set(positive_samples)
    remaining_samples_set = all_samples_set - positive_samples_set

    remaining_samples = []
    for circRNA_id, miRNA_id in tqdm(remaining_samples_set, desc="Processing Samples", unit="Sample"):
        circRNA_subcellular = circRNA_data[circRNA_id]
        miRNA_subcellular = miRNA_data[miRNA_id]

        if not any(any(subcellular in miRNA_subcellular for subcellular in circRNA_subcellular) for miRNA_subcellular in miRNA_subcellular):
            remaining_samples.append((circRNA_id, miRNA_id))

    num_negative_samples = len(positive_samples)
    negative_samples = random.sample(remaining_samples, num_negative_samples)

    return negative_samples

def save_negative_samples(negative_samples, output_file):

    df = pd.DataFrame(negative_samples, columns=['circRNA', 'miRNA'])
    df.to_csv(output_file, index=False, header=False)

if __name__ == "__main__":

    fasta_file = '../../dataset/9589/circRNA.fasta'
    xlsx_file = '../../dataset/9589/miRNA.xlsx'
    positive_samples_file = '../../dataset/9589/interaction.csv'
    negative_samples_file = 'NegativeSample.csv'

    circRNA_data = read_fasta(fasta_file)
    miRNA_data = read_xlsx(xlsx_file)
    positive_samples = read_positive_samples(positive_samples_file)

    negative_samples = generate_negative_samples(circRNA_data, miRNA_data, positive_samples)
    save_negative_samples(negative_samples, negative_samples_file)
