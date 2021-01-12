import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

def build_mean_std(df, base_dir):
    description = df.describe().to_dict()
    mean_std_dict = {}
    for key, value in description.items():
        mean_std_dict[key] = [value['mean'], value['std']]

    output_file_path = os.path.join(base_dir, 'mean_std.json')
    with open(output_file_path, mode="w") as output_file:
        json.dump(mean_std_dict, output_file)

def save_results(train_df, test_df, base_dir):
    train_df.to_csv(os.path.join(base_dir, 'train.csv'), index=False, header=True)
    test_df.to_csv(os.path.join(base_dir, 'test.csv'), index=False, header=True)

    build_vocab(train_df, base_dir)
    build_mean_std(train_df, base_dir)
    

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
columns = renameColumns.values()

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

save_results(train_df, test_df, base_dir)
