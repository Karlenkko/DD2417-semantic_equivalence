import codecs
import csv

import torch
import numpy as np
import pandas as pd



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    l = []
    with codecs.open("concatGRU_test_2_padding.csv", 'r', 'utf-8') as f:
        lines = csv.reader(f)
        is_first_line = True
        for line in lines:
            if is_first_line:
                is_first_line = False
                continue
            data = line[0][1:-2].split(",")
            l += [[float(data[0]), float(data[1])]]

    pd.DataFrame(l).to_csv('concatGRU_test_2_padding_out.csv', header=['concatGRU_1', 'concatGRU_2'], index=False)


