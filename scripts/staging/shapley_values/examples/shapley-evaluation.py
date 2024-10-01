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

# To debug the shap package, or output more information during computation, clone the git repo and add its path here.
# This will prepend the local installation of shap and makes sure it is imported by uncommenting the next block
'''
%load_ext autoreload
%autoreload 2
import os
import sys
module_path = os.path.abspath(os.path.join('<path-to-shap-repo>')) # or the path to your source code
sys.path.insert(0, module_path)
'''
# for command line args
import argparse
parser=argparse.ArgumentParser(description="Evaluate results of systemd shap.")
parser.add_argument("--data-path-x", default="../data/Adult_X.csv", help="Path to CSV with X data.")
parser.add_argument("--data-path-y", default="../data/Adult_Y.csv", help="Path to CSV with y data.")
parser.add_argument("--systemds-accuracy-path", default="../data/Adult_shap-values_10000smpl.csv", help="Path to shapley values computed with systemds.\n Make sure you used the same number of samples per feature as in this script.")
parser.add_argument("--systemds-times-path", default="../data/systemds_runtimes.csv", help="Path to runtimes collected from test_runtimes.sh.")
parser.add_argument("--python-runtimes-cache-path", default="../data/python_runtimes.csv", help="Path to cache runtimes computed by python cache.")
parser.add_argument("--recompute-python-runtimes", action='store_true', help="Don't use cached runtimes.")
parser.add_argument("--shap-method", choices=['sampling'], help="Currently only sampling is supported.")
parser.add_argument("--samples-per-feature", help="Number of samples per feature.", default=1000)
parser.add_argument("--no-accuracy-comparison", action='store_true')
parser.add_argument("--no-runtime-comparison", action='store_true')
parser.add_argument("--save-fig", action='store_true')
parser.add_argument("--log-level", choices=['10','20','30','40','50'], default='20', help="10 -> DEBUG, 20 -> INFO , 30 -> WARNING, 40 -> ERROR")
args=parser.parse_args()

import logging
logging.basicConfig(level=int(args.log_level))
logging.info("============ EVALUATION =============")
logging.info("Log Level: %s", int(args.log_level))
logging.info("Importing packages.")
# import packages
import pandas as pd
import numpy as np
import shap
import sklearn as sk
import matplotlib.pyplot as plt
import time
import os

#load prepared data into dataframe
logging.info("Loading Data from %s.", args.data_path_x)
df_x = pd.read_csv(args.data_path_x, header=None)
logging.info("Loading Data from %s.", args.data_path_y)
df_y = pd.read_csv(args.data_path_y, header=None)

#train model
logging.info("Training model with sklearn.")
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(df_x.values, df_y.values.ravel(), test_size=0.2, random_state=42)

model = sk.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')

model.fit(X_train, y_train)

#test model
y_pred = model.predict(X_test)
accuracy = sk.metrics.accuracy_score(y_test, y_pred)
conf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)

logging.debug("Accuracy: %s", accuracy)
logging.debug("Confusion Matrix:\n%s", conf_matrix)

#create SHAP sampling explainer
sampling_explainer = shap.explainers.SamplingExplainer(model.predict, df_x.values)

#define functions
def run_sampler_for(samples_per_feature):
    logging.debug("Running pyhton shap sampler for %s samples per feature.", samples_per_feature)
    start = time.time()
    # Using n samples for each feature (107*n in total) to match systemds
    shap_vals_sampling=sampling_explainer.explain(df_x.iloc[1:2].values, nsamples=107*samples_per_feature, min_samples_per_feature=samples_per_feature)
    end = time.time()

    total_t=end-start
    logging.debug("The time of execution of sampling explainer is %s seconds.", total_t)
    return shap_vals_sampling, total_t

def eval_and_plot_accuracy():
    logging.info("=== ACCURACY EVALUATION ===")
    logging.info("Running sampler for %s samples per feature.", args.samples_per_feature)
    shap_vals_sampling, _ = run_sampler_for(args.samples_per_feature)
    # load results from systemds implementation
    logging.info("Loading comparison data from '%s' .",args.systemds_accuracy_path)
    df_comp = pd.read_csv(args.systemds_accuracy_path, header=None, names=['systemds_shap'])

    # add results from SHAP package
    df_comp['python_shap']=shap_vals_sampling


    #plot
    logging.info("Creating comparison plot.")
    barWidth = 0.4  # Width of the bars
    positions = np.arange(len(df_comp))

    mse = sk.metrics.mean_squared_error(df_comp.python_shap, df_comp.systemds_shap)

    plt.figure(figsize=(12,6))
    plt.bar(positions-barWidth/2 , df_comp.systemds_shap, label="systemds", width=barWidth)
    plt.bar(positions+barWidth/2 , df_comp.python_shap, label="SHAP package", width=barWidth)

    plt.text(1, -0.24, 'MSE: '+str(mse))

    # Additional plot formatting
    plt.title('Shapley Values for the Adult Dataset (scaled)')
    plt.xlabel('Index of transformencoded features\n (truncated larger than 70, since they were mostly zero)')
    plt.ylabel('Shapley Value')
    plt.xlim(0,70)
    plt.legend()
    if args.save_fig:
        logging.info("Saving to accuracy_plot.png")
        plt.savefig('accuracy_plot.png')
    plt.show()

#runtime comaprisons
def run_and_store_runtimetests_for_python_shap():
    python_times=np.zeros((10*3,2))
    ind=0
    for i in range(10):
        samples = 5000*(i+1)
        logging.info("Running for %s samples.", samples)
        time_sum = 0
        for j in range(3):
            logging.debug("Iteration %s", j)
            _ , total_time = run_sampler_for(samples)
            python_times[ind,0]=samples
            python_times[ind,1]=total_time
            ind +=1

    python_times_df = pd.DataFrame(python_times)
    python_times_df.to_csv(args.python_runtimes_cache_path)

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0
def eval_and_plot_runtimes():
    logging.info("=== RUNTIMES EVALUATION ===")
    if (not is_non_zero_file(args.systemds_times_path)):
        logging.error("There is no file for systemds data at %s !\nPlease provide runtimes from systemDS by running test_runtimes.sh.", args.systemds_times_path)
        exit()

    if(is_non_zero_file(args.python_runtimes_cache_path) and not args.recompute_python_runtimes):
        logging.info("Found precomputed python runtimes at %s, reusing them.", args.python_runtimes_cache_path)
    else:
        logging.info("Computing python runtimes and caching them at %s", args.python_runtimes_cache_path)
        run_and_store_runtimetests_for_python_shap()

    logging.info("Loading runtimes from files.")
    logging.info("SystemDS runtimes from %s", args.systemds_times_path)
    logging.info("Python runtimes from %s", args.python_runtimes_cache_path)
    times_df=pd.read_csv(args.systemds_times_path)
    python_times_df=pd.read_csv(args.python_runtimes_cache_path)
    times_df['runtime python'] = python_times_df.iloc[:,2]
    logging.debug("Loaded Dataframe:\n%s", times_df)
    logging.info("Plotting comparison.")
    plt.plot(times_df.groupby('samples').mean().runtime, label="SystemDS")
    plt.plot(times_df.groupby('samples').mean()['runtime python'], label="Shap Python Package")
    plt.title("Runtimes of SystemDS vs Python Implementation")
    plt.xlabel("# Samples per Feature")
    plt.ylabel("Runtime is Seconds")
    plt.legend()
    if args.save_fig:
        logging.info("Saving to runtime_comparison_plot.png")
        plt.savefig('runtime_comparison_plot.png')
    plt.show()

if (not args.no_accuracy_comparison):
    logging.info("Computing and plotting accuracy comparison.")
    eval_and_plot_accuracy()

if (not args.no_runtime_comparison):
    logging.info("Computing and plotting runtime comparison.")
    eval_and_plot_runtimes()

