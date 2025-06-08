import pandas as pd

train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values

# print(train_data)
#
# print(test_data)

print(train_data.shape)
print(test_data.shape)