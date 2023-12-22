import pandas as pd
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_features(sequence):

    inputs = tokenizer.encode_plus(sequence, add_special_tokens=True, return_tensors="pt")
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :8]

circrna_df = pd.read_csv("../dataset/9905/circRNA.csv", header=None)

circrna_df.columns = ["id", "sequence"]
circrna_df["sequence"] = circrna_df["sequence"].apply(lambda x: x[:10])
circrna_df["vector"] = circrna_df["sequence"].apply(lambda x: extract_features(x))
circrna_df = pd.concat([circrna_df["id"], circrna_df["vector"].apply(pd.Series)], axis=1)
circrna_df.to_csv("circrna_vector.csv", index=False, header=False)

mirna_df = pd.read_csv("../dataset/9905/miRNA.csv", header=None)

mirna_df.columns = ["id", "sequence"]
mirna_df["sequence"] = mirna_df["sequence"].apply(lambda x: x[:10])

mirna_df["vector"] = mirna_df["sequence"].apply(lambda x: extract_features(x))

mirna_df = pd.concat([mirna_df["id"], mirna_df["vector"].apply(pd.Series)], axis=1)

mirna_df.to_csv("mirna_vector.csv", index=False, header=False)



circrna_df = pd.read_csv("9905/circrna_vector.csv", header=None)
mirna_df = pd.read_csv("9905/mirna_vector.csv", header=None)
merged_df = pd.concat([circrna_df, mirna_df], axis=0)
merged_df.to_csv("bert_vector.csv", index=False, header=False)

