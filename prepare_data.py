import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
#from face_detection import RetinaFace
from torchvision import datasets
import numpy as np
import random

def convert_object_type_to_category(df):
    """Converts columns of type object to category."""
    df = pd.concat(
        [
            df.select_dtypes(include=[], exclude=['object']),
            df.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
        ],
        axis=1).reindex(df.columns, axis=1)
    return df

def load_df(path, columns=None, skiprows=None):
    df = pd.read_csv(path, sep=',', names=columns, skiprows=skiprows, skipinitialspace=True)
    # Convert columns of type ``object`` to ``category`` 
    df = convert_object_type_to_category(df)

    return df

def apply_dict(df, dictionary, column):
    df[column] = df.apply(lambda x: dictionary[x[column]] if x[column] in dictionary.keys() else 'Other', axis=1).astype('category')

def apply_dicts(df, race_dict=None, sex_dict=None):
    if race_dict is not None:
        apply_dict(df, race_dict, 'race')
    if sex_dict is not None:
        apply_dict(df, sex_dict, 'sex')
    
def build_vocab(df, base_dir):
    cat_cols = df.select_dtypes(include='category').columns
    vocab_dict = {}
    for col in cat_cols:
        vocab_dict[col] = list(set(train_df[col].cat.categories))

    output_file_path = os.path.join(base_dir, 'vocabulary.json')
    with open(output_file_path, mode="w") as output_file:
        json.dump(vocab_dict, output_file)

def build_mean_std(df, base_dir, suffix=''):
    description = df.describe().to_dict()
    mean_std_dict = {}
    for key, value in description.items():
        print(key)
        print(value)
        mean_std_dict[key] = [value['mean'], value['std']]

    output_file_path = os.path.join(base_dir, f'mean_std{suffix}.json')
    with open(output_file_path, mode="w") as output_file:
        json.dump(mean_std_dict, output_file)

def save_results(train_df, test_df, base_dir, contains_numeric=True, suffix='', skip_vocab=False):
    train_df.to_csv(os.path.join(base_dir, f'train{suffix}.csv'), index=False, header=True)
    test_df.to_csv(os.path.join(base_dir, f'test{suffix}.csv'), index=False, header=True)

    if not skip_vocab:
        build_vocab(train_df, base_dir)
    if contains_numeric:
        build_mean_std(train_df, base_dir, suffix)
    

##########
# COMPAS #
##########
print("Processing COMPAS")
base_dir = "data/COMPAS"
filepath = os.path.join(base_dir, "compas-scores-two-years.csv")
columns = ['juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count',
           'age', 
           'c_charge_degree', 
           'c_charge_desc',
           'age_cat',
           'sex', 'race',  'is_recid']

df = load_df(filepath)
# only drop duplicates if they have the same ID:
df = df[['id'] + columns].drop_duplicates()
df = df[columns]

# Convert race to Black, White and Other
race_dict = {
    'African-American': 'Black',
    'Caucasian': 'White'
}
apply_dicts(df, race_dict=race_dict)

df['is_recid'] = df.apply(lambda x: 'Yes' if x['is_recid']==1.0 else 'No', axis=1).astype('category')

# There are a few entries where c_charge_degree is missing
# We just remove those
df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

save_results(train_df, test_df, base_dir)


########
# LSAC #
########
print("Processing LSAC")
base_dir = "data/LSAC"
filepath = os.path.join(base_dir, "lsac.sas7bdat")
df = pd.read_sas(filepath)
df = convert_object_type_to_category(df)

columns = ['zfygpa','zgpa','DOB_yr','parttime','gender','race','tier','fam_inc','lsat','ugpa','pass_bar','index6040']
df = df[columns]
renameColumns = {
    'gender': 'sex',
    'index6040': 'weighted_lsat_ugpa',
    'fam_inc': 'family_income',
    'tier': 'cluster_tier',
    'parttime': 'isPartTime'
}

# Renaming columns
df = df.rename(columns = renameColumns)
columns = list(df.columns)

# NaN in 'pass_bar' refer to dropouts. Considering NaN as failing the bar.
df['pass_bar'] = df['pass_bar'].fillna(value=0.0)
df['pass_bar'] = df.apply(lambda x: 'Passed' if x['pass_bar']==1.0 else 'Failed_or_not_attempted', axis=1).astype('category')

df['zfygpa'] = df['zfygpa'].fillna(value=0.0)
df['zgpa'] = df['zgpa'].fillna(value=0.0)
df['DOB_yr'] = df['DOB_yr'].fillna(value=0.0)
df = df.dropna()

# Binarize target_variable
df['isPartTime'] = df.apply(lambda x: 'Yes' if x['isPartTime']==1.0 else 'No', axis=1).astype('category')

# Process protected-column values
race_dict = {
    3.0:'Black',
    7.0: 'White'
}
sex_dict = {
    b'female': 'Female',
    b'male': 'Male'
}
apply_dicts(df, race_dict, sex_dict)

train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

save_results(train_df, test_df, base_dir)

#########
# Adult #
#########
print("Processing UCI Adult")
base_dir = "data/Adult"
train_file = os.path.join(base_dir, 'adult.data')
test_file = os.path.join(base_dir, 'adult.test')

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

train_df = load_df(train_file, columns=columns)
# The first line of the test file is not a datapoint
test_df = load_df(test_file, columns=columns, skiprows=1)

# Remove the dot in the income column
test_df['income'] = test_df['income'].apply(lambda x: x[:-1])

print('Frequencies sex in train:')
print(train_df.groupby('sex').count() / len(train_df))
print('Frequencies sex in test:')
print(test_df.groupby('sex').count() / len(test_df))

save_results(train_df, test_df, base_dir)

'''
### approach 1: throw away test file and re-split train file
all_idcs = np.random.permutation(np.arange(len(train_df)))
train_idcs, test_idcs = all_idcs[:int(len(all_idcs) * 0.7)], all_idcs[int(len(all_idcs) * 0.7):]
train_only_test_df = train_df.iloc[test_idcs,:]
train_only_train_df = train_df.iloc[train_idcs,:]
print('Frequencies sex in only_train train:')
print(train_only_train_df.groupby('sex').count()/len(train_only_train_df))
print('Frequencies sex in only_train test:')
print(train_only_test_df.groupby('sex').count()/len(train_only_test_df))

save_results(train_only_train_df, train_only_test_df, base_dir, suffix='_only_train', skip_vocab=True)
###

### approach 2: combine both files
concat_df = pd.concat([train_df, test_df])
all_idcs = np.random.permutation(np.arange(len(concat_df)))
train_idcs, test_idcs = all_idcs[:int(len(all_idcs) * 0.7)], all_idcs[int(len(all_idcs) * 0.7):]
concat_test_df = concat_df.iloc[test_idcs,:]
concat_train_df = concat_df.iloc[train_idcs,:]
print('Frequencies sex in train:')
print(concat_train_df.groupby('sex').count()/len(concat_train_df))
print('Frequencies sex in test:')
print(concat_test_df.groupby('sex').count()/len(concat_test_df))

save_results(concat_train_df, concat_test_df, base_dir, suffix='_concat', skip_vocab=True)
###

'''
############
# FairFace #
############
print("Processing FairFace")
base_dir = "data/FairFace/labels"
train_file = os.path.join(base_dir, 'train_original.csv')
test_file = os.path.join(base_dir, 'test_original.csv')

columns = [
    "file", "age", "gender", "race", "service_test"
]

train_df = load_df(train_file)
test_df = load_df(test_file)

# Remove the folder name in file path
train_df['file'] = train_df['file'].apply(lambda x: x[6:])
test_df['file'] = test_df['file'].apply(lambda x: x[4:])

# delete irrelevant columns
del train_df['service_test']
del test_df['service_test']
del train_df['age']
del test_df['age']

save_results(train_df, test_df, base_dir, contains_numeric=False)


####################
# FairFace Reduced #
####################
BASE_PATH = "data/FairFace"
BASE_PATH_NEW = "data/FairFace_reduced"
THRESHOLD = 0.987

for TEST in [False, True]:
    idx = []

    frame = pd.read_csv(os.path.join(BASE_PATH, "test.csv" if TEST else "train.csv"))
    detector = RetinaFace()

    # select images with high confidence that face is contained (prefers full front shots)
    for i in tqdm(range(len(frame))):
        img = Image.open(os.path.join(BASE_PATH, "images", "test" if TEST else "train", frame.iloc[i].file)).convert('RGB')
        np_img = np.array(img)

        try:
            # moved this into try/catch as detector is sometimes unstable
            all_faces = detector([np_img])
            box, landmarks, score = all_faces[0][0]

            if score > THRESHOLD:
                idx.append(i)
                resized_img = img.resize((80, 80), Image.ANTIALIAS)
                resized_img.save(os.path.join(BASE_PATH_NEW, "images", "test" if TEST else "train", frame.iloc[i].file), quality=60)

        except:
            pass


    sub_frame = frame.iloc[idx]
    sub_frame.to_csv(os.path.join(BASE_PATH_NEW, "test.csv" if TEST else "train.csv"), index=False)

#########
# MNIST #
#########
mnist_trainset = datasets.MNIST(root='data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='data', train=False, download=True, transform=None)

mnist_trainset_np = []
mnist_testset_np = []

for i in tqdm(range(len(mnist_trainset))):
    img, label = mnist_trainset.__getitem__(i)
    if (label == 8 or label == 5) and random.random() <= 0.6:
        sample = [np.array(img), label]
        mnist_trainset_np.append(sample)

for i in tqdm(range(len(mnist_testset))):
    img, label = mnist_testset.__getitem__(i)
    if (label == 8 or label == 5) and random.random() <= 0.6:
        sample = [np.array(img), label]
        mnist_testset_np.append(sample)

np.save(os.path.join('data', 'MNIST', 'mnist_trainset'), np.array(mnist_trainset_np, dtype=object))
np.save(os.path.join('data', 'MNIST', 'mnist_testset'), np.array(mnist_testset_np, dtype=object))

os.system('rm -r data/MNIST/processed')
os.system('rm -r data/MNIST/raw')

######################
# EMNIST with Colors #
######################
mnist_trainset = datasets.EMNIST(root='data', split='balanced', train=True, download=True, transform=None)
mnist_testset = datasets.EMNIST(root='data', split='balanced', train=False, download=True, transform=None)

for t in ['train', 'test']:
    dataset = mnist_trainset if t == 'train' else mnist_testset
    new_dataset = []
    for i in tqdm(range(len(dataset))):
        x, original_label = dataset.__getitem__(i)
        y = int(original_label >= 24)

        if (original_label % 2 == 0 and random.random() <= 0.9) or (original_label % 2 == 1 and random.random() <= 0.1):
            if random.random() <= 0.3:
                continue
            s = 1
        else:
            s = 0

        new_dataset.append([np.array(x), y, s])

    np.save(os.path.join("data", "EMNIST", t + '_prepared'), np.array(new_dataset))
