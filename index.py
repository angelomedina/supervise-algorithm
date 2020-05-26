import pandas as pd;
import numba as np;
import random;
import math;

def loadcsv(file):
    dataset = pd.read_csv(file)
    dataset = dataset.values
    return dataset

def split_dataset(dataset, split_ratio):
    train_size = int( len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]

def main():
    dataset = loadcsv("dataset.csv")
    split_ratio = 0.8
    train_set, test_set = split_dataset(dataset, split_ratio)
    print(test_set[0])

if __name__ == "__main__":
    main()    