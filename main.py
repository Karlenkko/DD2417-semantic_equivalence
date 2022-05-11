import torch
import numpy as np
import pandas as pd



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    training = pd.read_csv("training.csv")
    data = training["question1"].tolist()
    print(np.shape(data))
    print(data[0])

    data = training["question2"].tolist()
    print(data[1])
    test = pd.read_csv("test.csv")
    print(np.shape(test))
    data = test["question2"].tolist()
    print(data[23])

