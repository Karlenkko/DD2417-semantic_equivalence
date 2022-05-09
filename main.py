import torch
import numpy as np
import pandas as pd



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "/content/gdrive/MyDrive/Semantic Equivalence Detection/questions.csv"
    data = pd.read_csv(file_path,
                       dtype={'id': np.int32, 'qid1': np.int32, 'qid2': np.int32, 'question1': str, 'question2': str,
                              'is_duplicate': np.int8})
    data = data[["question1", "question2", "is_duplicate"]]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
