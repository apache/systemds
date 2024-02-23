#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

#%%



import pandas as pd
import shap
import sklearn as sk
import time
import os
import datetime

# for command line args
import argparse
parser=argparse.ArgumentParser(description="Run permutation shap and time it.")
parser.add_argument("--data-path-x", default="../data/Adult_X.csv", help="Path to CSV with X data.")
parser.add_argument("--data-path-y", default="../data/Adult_Y.csv", help="Path to CSV with y data.")
parser.add_argument("--result-path", default="../data/python_shap_permutation.csv", help="Path to append results to.")
parser.add_argument("--n-instances", help="Number of instances.", default=1)
parser.add_argument("--n-permutations", help="Number of permutations.", default=1)
parser.add_argument('--silent', action='store_true', help='Don\'t print a thing.')
parser.add_argument('--just-print-t', action='store_true', help='Don\'t store, just print time at end.')
args=parser.parse_args()


#%%
#load prepared data into dataframe

df_x = pd.read_csv(args.data_path_x, header=None)
df_y = pd.read_csv(args.data_path_y, header=None)
#%%
#train model
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(df_x.values, df_y.values.ravel(), test_size=0.2, random_state=42)

model = sk.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')

model.fit(X_train, y_train)
#%%
#test model
y_pred = model.predict(X_test)
accuracy = sk.metrics.accuracy_score(y_test, y_pred)
conf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)

if not args.silent:
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
#%%
#create SHAP  explainer
#sampling_explainer = shap.explainers.SamplingExplainer(model.predict, df_x.values)
if not args.silent:
    print(int(args.n_permutations))
start_exp = time.time()
permutation_explainer = shap.explainers.Permutation(model.predict_proba, df_x.values)
shap_values = permutation_explainer(df_x.iloc[1:1+int(args.n_instances)],
                                    max_evals=2*len(df_x.iloc[1])*(int(args.n_permutations)+1))
end_exp = time.time()
total_t=end_exp-start_exp

if not args.silent:
    print("Time:", total_t, "s")
#%%
filename=args.result_path
data = {
    'recorded_at': [datetime.datetime.now()],
    'num_instances': [args.n_instances],
    'runtime_seconds': [total_t],
}
if args.just_print_t:
    print(str(total_t))
else:
    df_times=pd.DataFrame(data)
    df_times.to_csv(filename, mode='a', header=not os.path.exists(filename))