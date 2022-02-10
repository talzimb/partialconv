import pandas as pd
import os

def read_df(path, type):
    '''This function reads train/ test txt files and remove unnecessary columns'''

    df = pd.read_csv(os.path.join(path, type), sep=" ", header=None)
    df.columns = ['patient id', 'filename', 'class', 'data source']
    df = df.drop(['patient id', 'data source'], axis=1)

    return df