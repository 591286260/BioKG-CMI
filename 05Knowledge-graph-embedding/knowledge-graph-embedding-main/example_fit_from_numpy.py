import random
import pandas as pd
import numpy as np
import tensorflow as tf
from KGE.data_utils import index_kg, convert_kg_to_index
from KGE.models.translating_based.TransE import TransE
import os

if __name__ == "__main__":
    train = np.loadtxt("./data/fb15k/train/train.csv", dtype=str, delimiter=',')
    valid = np.loadtxt("./data/fb15k/valid/valid.csv", dtype=str, delimiter=',')
    test = np.loadtxt("./data/fb15k/test/test.csv", dtype=str, delimiter=',')

    metadata = index_kg(train)
    metadata["ind2type"] = random.choices(["A", "B", "C"], k=len(metadata["ind2ent"]))

    train = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
    valid = convert_kg_to_index(valid, metadata["ent2ind"], metadata["rel2ind"])
    test = convert_kg_to_index(test, metadata["ent2ind"], metadata["rel2ind"])

    model = TransE(
        embedding_params={"embedding_size": 3},
        negative_ratio=4,
        corrupt_side="h+t"
    )

    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=2, batch_size=512,
                early_stopping_rounds=None, restore_best_weight=False,
                optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                seed=12345, log_path="tensorboard_logs", log_projector=True)

    eval_result_filtered = model.evaluate(eval_X=test, corrupt_side="h",
                                          positive_X=np.concatenate((train, valid, test), axis=0))
    print(eval_result_filtered)

    ###########分割线
    # 获取实体和关系的嵌入向量
    entity_ids = np.array(metadata["ind2ent"])  # 转换为NumPy数组
    relation_ids = np.array(metadata["ind2rel"])  # 转换为NumPy数组
    entity_embeddings = model.model_weights["ent_emb"].numpy()
    relation_embeddings = model.model_weights["rel_emb"].numpy()

    # # 将嵌入向量转换为 pandas DataFrame
    # entity_df = pd.DataFrame(entity_embeddings, columns=[f"entity_dim_{i}" for i in range(entity_embeddings.shape[1])])
    # relation_df = pd.DataFrame(relation_embeddings,
    #                            columns=[f"relation_dim_{i}" for i in range(relation_embeddings.shape[1])])

    # 将嵌入向量和ID转换为 pandas DataFrame
    entity_df = pd.DataFrame(np.hstack((entity_ids.reshape(-1, 1), entity_embeddings)),
                             columns=["entity_id"] + [f"entity_dim_{i}" for i in range(entity_embeddings.shape[1])])
    relation_df = pd.DataFrame(np.hstack((relation_ids.reshape(-1, 1), relation_embeddings)),
                               columns=["relation_id"] + [f"relation_dim_{i}" for i in
                                                          range(relation_embeddings.shape[1])])



    # 定义保存路径
    save_dir = "embedding_vectors_csv"
    os.makedirs(save_dir, exist_ok=True)

    # 保存实体和关系的嵌入向量为CSV文件
    entity_csv_file = os.path.join(save_dir, "entity_embeddings.csv")
    relation_csv_file = os.path.join(save_dir, "relation_embeddings.csv")

    entity_df.to_csv(entity_csv_file, index=False)
    relation_df.to_csv(relation_csv_file, index=False)

    print("Entity embeddings saved to:", entity_csv_file)
    print("Relation embeddings saved to:", relation_csv_file)
