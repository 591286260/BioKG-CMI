import random
import pandas as pd
import numpy as np
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.semantic_based.DistMult import DistMult
import os

if __name__ == "__main__":
    train = np.loadtxt("data/755/interaction.csv", dtype=str, delimiter=',', encoding='gbk')

    metadata = index_kg(train)
    metadata["ind2type"] = random.choices(["A", "B", "C"], k=len(metadata["ind2ent"]))

    train = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])

    model = DistMult(
        embedding_params={"embedding_size": 128},
        corrupt_side="h+t"
    )

    val_X = np.zeros_like(train)

    model.train(train_X=train, val_X=val_X, metadata=metadata, epochs=2, batch_size=128,
                early_stopping_rounds=None, restore_best_weight=False,
                optimizer=tf.optimizers.Adam(learning_rate=0.001),
                seed=12345, log_path="tensorboard_logs", log_projector=True)


    entity_ids = np.array(metadata["ind2ent"])
    entity_embeddings = model.model_weights["ent_emb"].numpy()


    entity_df = pd.DataFrame(np.hstack((entity_ids.reshape(-1, 1), entity_embeddings)),
                             columns=["entity_id"] + [f"entity_dim_{i}" for i in range(entity_embeddings.shape[1])])


    save_dir = "data/9589/embedding_vectors_csv"
    os.makedirs(save_dir, exist_ok=True)

    entity_csv_file = os.path.join(save_dir, "entity_embeddings.csv")

    entity_df.to_csv(entity_csv_file, index=False, header=False)
