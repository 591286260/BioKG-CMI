import csv
import numpy as np

def read_my_csv(save_list, file_name):
    with open(file_name, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            save_list.append(row)

def store_file(data, file_name):
    with open(file_name, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)

data1 = []
read_my_csv(data1, "../../dataset/9589/interaction.csv")
print(len(data1))

data2 = []
read_my_csv(data2, "../../01Negative sample generation/NegativeSample.csv")
print(len(data2))

data_final = np.vstack((data1, data2))
print(data_final.shape)
print(data_final)

store_file(data_final, 'Positive_and_negative_samples.csv')
