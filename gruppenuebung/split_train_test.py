import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('online_shoppers_intention.csv')

# transform all non-numeric data into numeric data
dataset = dataset.apply(preprocessing.LabelEncoder().fit_transform)

# select the last column as label (y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

train, test = train_test_split(dataset, test_size=1 / 3, random_state=0, stratify=y)
train.to_csv('online_shoppers_intention_train.csv', index=False)
test.to_csv('online_shoppers_intention_test.csv', index=False)