import pandas as pd
import torch
from torch.nn.functional import pairwise_distance
from gensim.models import Word2Vec
from tqdm import tqdm

circRNA_data = pd.read_csv('../dataset/9905/circRNA.csv', header=None)
circRNA_ids = circRNA_data.iloc[:, 0].tolist()
circRNA_sequences = [seq.split() for seq in circRNA_data.iloc[:, 1].tolist()]

miRNA_data = pd.read_csv('../dataset/9905/miRNA.csv', header=None)
miRNA_ids = miRNA_data.iloc[:, 0].tolist()
miRNA_sequences = [seq.split() for seq in miRNA_data.iloc[:, 1].tolist()]

word2vec_model = Word2Vec(sentences=circRNA_sequences + miRNA_sequences, vector_size=64, window=7, min_count=1, workers=4)

word2vec_model.save('path_to_word2vec_model.bin')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word2vec_model = Word2Vec.load('path_to_word2vec_model.bin')
word2vec_model_vectors = torch.tensor(word2vec_model.wv.vectors).to(device)

circRNA_sim_matrix = []
for seq1 in tqdm(circRNA_sequences, desc="Calculating circRNA similarities"):
    seq1_tokens = torch.tensor([word2vec_model.wv.key_to_index[token] for token in seq1 if token in word2vec_model.wv]).to(device)
    seq1_embedding = torch.mean(word2vec_model_vectors[seq1_tokens], dim=0, keepdim=True)

    sim_row = []
    for seq2 in circRNA_sequences:
        seq2_tokens = torch.tensor([word2vec_model.wv.key_to_index[token] for token in seq2 if token in word2vec_model.wv]).to(device)
        seq2_embedding = torch.mean(word2vec_model_vectors[seq2_tokens], dim=0, keepdim=True)

        distance = pairwise_distance(seq1_embedding, seq2_embedding)
        sim_row.append(distance.item())

    circRNA_sim_matrix.append(sim_row)

circRNA_sim_df = pd.DataFrame(circRNA_sim_matrix, index=circRNA_ids, columns=circRNA_ids)



