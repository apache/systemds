import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import datetime
from joblib import dump, load

# for command line args
import argparse
parser=argparse.ArgumentParser(description="Prepare Models for permutation shap experiments.")
parser.add_argument("--data-dir", default="../data/adult/", help="Path to CSV with X data.")
parser.add_argument("--data-x", default="Adult_X.csv", help="Path to CSV with X data.")
parser.add_argument("--data-y", default="Adult_y.csv", help="Path to CSV with y data.")
parser.add_argument("--model-type", default="multiLogReg", help="Model type to prepare.")
args=parser.parse_args()

#load prepared data into dataframe

df_x = pd.read_csv(args.data_dir+args.data_x, header=None)
df_y = pd.read_csv(args.data_dir+args.data_y, header=None)
X_train, X_test, y_train, y_test = train_test_split(df_x.values, df_y.values.ravel(), test_size=0.2, random_state=42)

if args.model_type == "multiLogReg":

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    model.fit(X_train, y_train)

    #test model
    y_pred = model.predict(X_test)
    accuracy = sk.metrics.accuracy_score(y_test, y_pred)
    conf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)


    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")

if args.model_type == "l2svm":

    model = SVC(kernel='linear', decision_function_shape='ovo')
    model.fit(X_train, y_train)

    #test model
    y_pred = model.predict(X_test)
    accuracy = sk.metrics.accuracy_score(y_test, y_pred)
    conf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)


    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")


#safe model to disk
dump(model, args.data_dir+args.model_type+".joblib")
