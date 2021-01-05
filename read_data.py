import numpy as np
import pandas as pd

path = './data/uci_adult/adult.data'
test_path = './data/uci_adult/adult.test'

data = pd.read_csv(path, delimiter=',')
print(data)
print(data.shape)

test_data = pd.read_csv(test_path, delimiter=',')
print(test_data)
print(test_data.shape)
print(len(test_data))

