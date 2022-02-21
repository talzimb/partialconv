import pandas as pd
import os

def read_df(path, type):
    '''This function reads train/ test txt files and remove unnecessary columns'''

    df = pd.read_csv(os.path.join(path, type), sep=" ", header=None)
    df.columns = ['patient id', 'filename', 'class', 'data source']
    df = df.drop(['patient id', 'data source'], axis=1)

    return df

def file_paths():
    '''create txt file with all image path, mask path and class in the selected folder '''
    root = r'/home/projects/yonina/SAMPL_training/covid_partialconv/train'
    train_df = read_df('/home/projects/yonina/SAMPL_training/covid_partialconv/', 'train.txt')
    paths = []
    masks = []
    classes = []
    for i in range(len(os.listdir(root))):
        filename = os.listdir(root)[i]
        if filename =='Thumbs.db': continue
        paths.append(os.path.join(root, filename))
        maskname =filename.rsplit('.', 1)[0] + '_mask.png'
        masks.append(os.path.join(root + '_masks', maskname))
        classes.append((train_df['class'][train_df['filename'] == filename]).values[0])
    with open(os.path.join(root, 'paths.txt'), 'w') as f:
        for item, mask, cls in zip(paths, masks, classes):
            f.write(f'{item} {mask} {cls}\n')

