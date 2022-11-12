from preprocessing import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns



def main():
    DATA_DIR = "../data/"
    data = pd.read_csv(os.path.join(DATA_DIR,os.listdir(DATA_DIR)[0]),index_col=["Unnamed: 0"])

    df = preprocessing(data,'text')

    print(df.head())


if __name__ == '__main__':
    main()
