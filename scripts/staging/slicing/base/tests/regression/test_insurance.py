import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import slicing.slicer as slicer

file_name = 'insurance.csv'
dataset = pd.read_csv(file_name)
attributes_amount = len(dataset.values[0])
# for now working with regression datasets, assuming that target attribute is the last one
# currently non-categorical features are not supported and should be binned
y = dataset.iloc[:, attributes_amount - 1:attributes_amount].values
# starting with one not including id field
x = dataset.iloc[:, 0:attributes_amount - 1].values
# list of numerical columns
non_categorical = [1, 3]
for row in x:
    for attribute in non_categorical:
        # <attribute - 2> as we already excluded from x id column
        row[attribute - 1] = int(row[attribute - 1] / 5)
# hot encoding of categorical features
enc = OneHotEncoder(handle_unknown='ignore')
x = enc.fit_transform(x).toarray()
complete_x = []
complete_y = []
counter = 0
all_features = enc.get_feature_names()
# train model on a whole dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
for item in x_test:
    complete_x.append((counter, item))
    complete_y.append((counter, y_test[counter]))
    counter = counter + 1
x_size = counter
model = LinearRegression()
model.fit(x_train, y_train)
preds = (model.predict(x_test) - y_test) ** 2
f_l2 = sum(preds)/x_size
errors = []
counter = 0
for pred in preds:
    errors.append((counter, pred))
    counter = counter + 1
# alpha is size significance coefficient
# verbose option is for returning debug info while creating slices and printing it
# k is number of top-slices we want
# w is a weight of error function significance (1 - w) is a size significance propagated into optimization function
slicer.process(all_features, model, complete_x, f_l2, x_size, y_test, errors, debug=True, alpha=5, k=10,
               w=0.5, loss_type=0)
