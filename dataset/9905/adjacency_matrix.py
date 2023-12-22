import csv
import numpy as np

def generate_adjacency_matrix(csv_file1, csv_file2):
    # 读取第一个 CSV 文件中的 ID
    with open(csv_file1, 'r') as file1:
        reader1 = csv.reader(file1)
        ids1 = [row[0] for row in reader1]

    # 读取第二个 CSV 文件中的信息
    with open(csv_file2, 'r') as file2:
        reader2 = csv.reader(file2)
        rows2 = list(reader2)

    # 获取节点列表和边列表
    nodes = ids1  # 使用第一个 CSV 文件中的 ID 作为节点列表
    edges = [(row[0], row[1]) for row in rows2 if row[0] in ids1 and row[1] in ids1]

    # 创建邻接矩阵
    num_nodes = len(nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 填充邻接矩阵
    for edge in edges:
        source = nodes.index(edge[0])
        target = nodes.index(edge[1])
        adjacency_matrix[source, target] = 1
        adjacency_matrix[target, source] = 1

    return adjacency_matrix

def save_adjacency_matrix_to_csv(adjacency_matrix, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adjacency_matrix)

# 示例用法
csv_file1 = 'id.csv'  # 包含 ID 的 CSV 文件路径
csv_file2 = 'interaction.csv'  # 包含图信息的 CSV 文件路径
adjacency_matrix = generate_adjacency_matrix(csv_file1, csv_file2)

output_csv_file = 'adjacency_matrix.csv'  # 输出的邻接矩阵 CSV 文件路径
save_adjacency_matrix_to_csv(adjacency_matrix, output_csv_file)
