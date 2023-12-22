import numpy as np
np.random.seed(1337)
import csv

def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

OriginalData = []
ReadMyCsv(OriginalData, "../../../dataset/9905/interaction.csv")
print(len(OriginalData))

LncDisease = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    LncDisease.append(Pair)
    counter = counter + 1

print('LncDisease', len(LncDisease))
print('OriginalData', len(OriginalData))

AllDisease = []
counter1 = 0
while counter1 < len(OriginalData):
    counter2 = 0
    flag = 0
    while counter2 < len(AllDisease):
        if OriginalData[counter1][1] != AllDisease[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllDisease[counter2]:
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllDisease.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllDisease)', len(AllDisease))

AllDRUG = []
counter1 = 0
while counter1 < len(OriginalData):
    counter2 = 0
    flag = 0
    while counter2 < len(AllDRUG):
        if OriginalData[counter1][0] != AllDRUG[counter2]:
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllDRUG[counter2]:
            flag = 1
            break
    if flag == 0:
        AllDRUG.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
print('len(AllDRUG)', len(AllDRUG))
storFile(AllDRUG, 'AllDRUG.csv')

PositiveSample = []

print('PositiveSample)', len(PositiveSample))

while counterN < len(PositiveSample):
    counterD = random.randint(0, len(AllDisease)-1)
    counterR = random.randint(0, len(AllDRUG)-1)
    DiseaseAndRnaPair = []
    DiseaseAndRnaPair.append(AllDRUG[counterR])
    DiseaseAndRnaPair.append(AllDisease[counterD])
    flag1 = 0
    counter = 0
    while counter < len(LncDisease):
        if DiseaseAndRnaPair == LncDisease[counter]:
            flag1 = 1
            break
        counter = counter + 1
    if flag1 == 1:
        continue
    flag2 = 0
    counter1 = 0
    while counter1 < len(NegativeSample):
        if DiseaseAndRnaPair == NegativeSample[counter1]:
            flag2 = 1
            break
        counter1 = counter1 + 1
    if flag2 == 1:
        continue
    if (flag1 == 0 & flag2 == 0):
        NegativePair = []
        NegativePair.append(AllDRUG[counterR])
        NegativePair.append(AllDisease[counterD])
        NegativeSample.append(NegativePair)
        counterN = counterN + 1
print('len(NegativeSample)', len(NegativeSample))
storFile(NegativeSample, 'NegativeSample.csv')