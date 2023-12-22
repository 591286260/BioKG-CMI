import csv
import pandas as pd

df1 = pd.read_csv('../../03Sequence feature extraction/9905/bert_vector.csv', header=None, index_col=0, encoding='gbk')
df2 = pd.read_csv('../../04Spatial proximity/9905/spatial proximity.csv', header=None, index_col=0, encoding='gbk')
df3 = pd.read_csv('/9905//knowledge-graph-embedding.csv', header=None, index_col=0, encoding='gbk')
merged_df = pd.concat([df1, df2, df3], axis=1)
merged_df.to_csv('./9905//BioKG-CMI.csv', header=None, encoding='gbk')

def read_feature_file(file_name):
    feature_dict = {}
    with open(file_name, 'r', encoding='gbk') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            id_value = row[0]
            feature_value = row[1:]
            feature_dict[id_value] = feature_value
    return feature_dict

def replace_id_with_feature(input_file_name, feature_dict, output_file_name):
    with open(input_file_name, 'r', encoding='gbk') as csvfile:
        with open(output_file_name, 'w', newline='', encoding='gbk') as output_csvfile:
            csv_writer = csv.writer(output_csvfile)

            for row in csv.reader(csvfile):
                id1, id2 = row
                feature1 = feature_dict.get(id1, [])
                feature2 = feature_dict.get(id2, [])
                combined_feature = feature1 + feature2
                csv_writer.writerow(combined_feature)

if __name__ == "__main__":

    feature_file_name = "BioKG-CMI.csv"

    feature_dict = read_feature_file(feature_file_name)

    input_file_name = "Positive and negative samples.csv"

    output_file_name = "./SampleFeature(BioKG-CMI).csv"
    replace_id_with_feature(input_file_name, feature_dict, output_file_name)
