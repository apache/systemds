#!/usr/bin/env python3
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

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from systemds.context import SystemDSContext
from systemds.operator.algorithm import als, lm, lmPredict

# To run this code, first download the MovieLens 100k dataset from 
# https://grouplens.org/datasets/movielens/100k/ and extract it to the specified folder.

data_folder = '/movie_data/ml-100k/'

def read_movie_data(n_rows: int=10000) -> pd.DataFrame:
    """
    Reads the MovieLens 100k dataset and returns a DataFrame with the following columns: user_id, item_id, rating.
    
    :param n_rows: Number of rows to read from the dataset.
    :return: DataFrame containing the movie ratings data.
    """

    # Load the MovieLens 100k dataset
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(data_folder + 'u.data', sep='\t', names=header)

    # Drop timestamp column
    ratings_df = ratings_df.drop('timestamp', axis=1)

    # Only check first n_rows rows to speed up processing
    ratings_df = ratings_df.head(n_rows)

    return ratings_df


def create_pivot_table(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pivot table from the ratings DataFrame where rows are users, columns are items, and values are ratings.

    :param ratings_df: DataFrame containing the movie ratings data with columns user_id, item_id, rating.
    :return: Pivot table with users as rows, items as columns, and ratings as values.
    """

    return ratings_df.pivot(index='user_id', columns='item_id', values='rating')


### Cosine Similarity Functions ###

def numpy_cosine_similarity(pivot_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Calculates the cosine similarity between users using NumPy.
    
    :param pivot_df: DataFrame containing the pivot table of user-item ratings.
    :return: DataFrame containing the cosine similarity between users and time taken.
    """

    # zeros = unrated items
    X = pivot_df.fillna(0).values

    start = time.time()

    # L2 norm of each user vector
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    # Normalize user vectors
    X_norm = X / norms

    # Cosine similarity = dot product of normalized vectors
    user_similarity = X_norm @ X_norm.T

    end = time.time()

    # convert to DataFrame for better readability
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=pivot_df.index,
        columns=pivot_df.index
    )

    return user_similarity_df, end - start


def systemds_cosine_similarity(pivot_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Calculates the cosine similarity between users using SystemDS.
    
    :param pivot_df: DataFrame containing the pivot table of user-item ratings.
    :return: DataFrame containing the cosine similarity between users and time taken.
    """

    # Zeros = unrated items
    X_np = pivot_df.fillna(0).values

    with SystemDSContext() as sds:

        start = time.time()

        # Load into SystemDS
        X = sds.from_numpy(X_np)

        # Compute L2 norms 
        row_sums = (X * X).sum(axis=1)
        norms = row_sums.sqrt()

        # Normalize user vectors
        X_norm = X / norms

        # Cosine similarity = dot product of normalized vectors
        user_similarity = X_norm @ X_norm.t()

        # Compute result
        user_similarity = user_similarity.compute()

        end = time.time()

        # Convert to DataFrame for better readability
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=pivot_df.index,
            columns=pivot_df.index
        )

    return user_similarity_df, end - start


def evaluate_cosine_similarity() -> None:
    """
    Evaluates and compares the cosine similarity computations between NumPy and SystemDS.
    """

    ratings = read_movie_data(100000)
    pivot_df = create_pivot_table(ratings)

    numpy_df, numpy_time = numpy_cosine_similarity(pivot_df)
    systemds_df, systemds_time = systemds_cosine_similarity(pivot_df)

    # Check if the results are approximately equal
    if np.allclose(numpy_df.values, systemds_df.values, atol=1e-8):
        print("Cosine similarity DataFrames are approximately equal.")
    else:
        print("Cosine similarity DataFrames are NOT equal.")

    print(f"Time taken for NumPy cosine similarity: {numpy_time}")
    print(f"Time taken for SystemDS cosine similarity: {systemds_time}")


### Matrix Factorization Functions ###

def numpy_als(pivot_df: pd.DataFrame, rank: int, reg: float, maxi: int) -> tuple[pd.DataFrame, float]:
    """
    Calculates a matrix R_hat using Alternating Least Squares (ALS) matrix factorization in numpy.
    
    :param pivot_df: DataFrame containing the pivot table of user-item ratings.
    :return: DataFrame containing the predicted ratings and time taken.
    """

    # Fill NaNs with zeros for computation
    R = pivot_df.fillna(0).values

    start = time.time()
    num_users, num_items = R.shape
    mask = (R != 0)

    # Random initialization of user and item factors
    P = np.random.rand(num_users, rank) * 0.01
    Q = np.random.rand(num_items, rank) * 0.01

    for iteration in range(maxi):

        # Update user factors
        for u in range(num_users):

            # Get only items user 'u' actually rated
            user_mask = mask[u, :] 
            Q_u = Q[user_mask, :]
            R_u = R[u, user_mask]
            
            if Q_u.shape[0] > 0:
                P[u, :] = np.linalg.solve(np.dot(Q_u.T, Q_u) + reg * np.eye(rank), np.dot(Q_u.T, R_u))

        # Update item factors
        for i in range(num_items):

            # Get only users who actually rated item 'i'
            item_mask = mask[:, i]
            P_i = P[item_mask, :]
            R_i = R[item_mask, i]
            
            if P_i.shape[0] > 0:
                Q[i, :] = np.linalg.solve(np.dot(P_i.T, P_i) + reg * np.eye(rank), np.dot(P_i.T, R_i))

    end = time.time()

    # Multiply P and Q to get the approximated ratings matrix
    R_hat = P @ Q.T

    # Convert to DataFrame for better readability
    ratings_hat_df = pd.DataFrame(
        R_hat,
        index=pivot_df.index,
        columns=pivot_df.columns
    )

    return ratings_hat_df, end - start


def systemds_als(pivot_df: pd.DataFrame, rank: int, reg: float, maxi: int) -> tuple[pd.DataFrame, float]:
    """
    Calculates a matrix R_hat using Alternating Least Squares (ALS) matrix factorization in SystemDS.
    
    :param pivot_df: DataFrame containing the pivot table of user-item ratings.
    :return: DataFrame containing the predicted ratings and time taken.
    """

    start = time.time()

    with SystemDSContext() as sds:

        # Load data into SystemDS
        R = sds.from_numpy(pivot_df.fillna(0).values)

        # Approximate factorization of R into two matrices P and Q using ALS
        P, Q = als(R, rank=rank, reg=reg, maxi=maxi).compute()
        end = time.time()

        # Multiply P and Q to get the approximated ratings matrix
        R_hat = P @ Q

    # Convert to DataFrame for better readability
    ratings_hat_df = pd.DataFrame(
        R_hat,
        index=pivot_df.index,
        columns=pivot_df.columns
    )

    return ratings_hat_df, end - start


def evaluate_als(model: str = "systemds", rank: int = 10, reg: float = 1.0, maxi: int = 20) -> None:
    """
    Evaluates and compares the ALS computations between NumPy and SystemDS. The data is split into training 
    and test sets with an 80/20 ratio. Then the RMSE is calculated for both sets.
    
    :param model: Model to use for ALS computation ("systemds" or "numpy").
    :param rank: Rank of the factorized matrices.
    :param reg: Regularization parameter.
    :param maxi: Maximum number of iterations.
    """

    ratings = read_movie_data(100000)
    pivot_df = create_pivot_table(ratings[:80000])

    if model == "systemds":
        ratings_hat_df, systemds_time = systemds_als(pivot_df, rank, reg, maxi)
    else:
        ratings_hat_df, numpy_time = numpy_als(pivot_df, rank, reg, maxi)

    # Print time taken
    print(f"Time taken for {model} ALS: ", systemds_time if model == "systemds" else numpy_time)

    # Training error
    mask = ~np.isnan(pivot_df.values)
    train_rmse = np.sqrt(np.mean((ratings_hat_df.values[mask] - pivot_df.values[mask])**2))
    print(f"Train RMSE for model with {model}: {train_rmse}")

    # Test error
    test_set = ratings[80000:]
    stacked_series = ratings_hat_df.stack()
    ratings_hat_long = stacked_series.reset_index()
    ratings_hat_long.columns = ['user_id', 'item_id', 'rating']

    merged_df = pd.merge(
        test_set, 
        ratings_hat_long, 
        on=['user_id', 'item_id'], 
        how='inner', 
        suffixes=('_actual', '_predicted')
    )

    # Force predictions to stay between 0.5 and 5.0
    merged_df['rating_predicted'] = merged_df['rating_predicted'].clip(0.5, 5.0)

    # Calculate root mean squared error (RMSE)
    squared_errors = (merged_df['rating_actual'] - merged_df['rating_predicted'])**2
    mse = np.mean(squared_errors)
    test_rmse = np.sqrt(mse)

    print(f"Test RMSE for model with {model}: {test_rmse}")


### Linear Regression ###

def preprocess_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function reads and preprocesses the MovieLens 100k dataset for linear regression. It returns four
    different numpy arrays: X_train, y_train, X_test, y_test. The preprocessing steps include:
    - Reading the datasets
    - Handling categorical variables
    - Normalizing numerical features
    - Merging datasets
    - Dropping unnecessary columns
    - Dropping rows with NaN values
    - Splitting into training and testing sets.

    :return: tuple of numpy arrays (X_train, y_train, X_test, y_test)
    """

    # Read datasets
    ratings_df = pd.read_csv(data_folder + 'u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    user_df = pd.read_csv(data_folder + 'u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    item_df = pd.read_csv(data_folder + 'u.item', sep='|', names=[
        'item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
        "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')
    
    # Turn categorical data into numerical
    user_df['gender'] = user_df['gender'].apply(lambda x: 0 if x == 'F' else 1)
    user_df = pd.get_dummies(user_df, columns=['occupation'])
    item_df['release_date'] = pd.to_datetime(item_df['release_date'], errors='raise', format='%d-%b-%Y')
    item_df['release_year'] = item_df['release_date'].dt.year

    # Normalize data
    user_df['age'] = (user_df['age'] - user_df['age'].min()) / (user_df['age'].max() - user_df['age'].min())
    item_df['release_year'] = (item_df['release_year'] - item_df['release_year'].min()) / (item_df['release_year'].max() - item_df['release_year'].min())

    # Merge datasets
    merged_df = ratings_df.merge(user_df, on='user_id').merge(item_df, on='item_id')

    # Drop unnecessary columns
    merged_df = merged_df.drop(['user_id', 'item_id', 'timestamp', 'zip_code', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'], axis=1)

    # Convert boolean columns to integers (important for NumPy and SystemDS)
    bool_cols = merged_df.select_dtypes(include=['bool']).columns
    merged_df[bool_cols] = merged_df[bool_cols].astype(int)

    # Drop rows with NaN values
    merged_df = merged_df.dropna()

    ratings = merged_df.pop('rating')
    features = merged_df

    # Split into train and test sets and convert to numpy arrays
    train_size = int(0.8 * len(ratings))
    X_train = features[:train_size].to_numpy()
    y_train = ratings[:train_size].to_numpy()
    X_test = features[train_size:].to_numpy()
    y_test = ratings[train_size:].to_numpy()

    print("NaNs in X:", np.isnan(X_train).any())
    print("NaNs in y:", np.isnan(y_train).any())

    return X_train, y_train, X_test, y_test


def linear_regression_pytorch(X_train, y_train, X_test, y_test, num_epochs=1000) -> tuple[float, float]:
    """
    Trains a linear regression model using PyTorch.

    :param X_train, X_test: numpy arrays of shape (n_samples, n_features)
    :param y_train, y_test: numpy arrays of shape (n_samples,)
    :param num_epochs: number of training iterations

    :return rmse: RMSE on test set
    :return time taken: time in seconds for training and prediction
    """

    start = time.time()

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().reshape(-1, 1)
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float().reshape(-1, 1)

    # Define model
    n_features = X_train.shape[1]
    class linearRegression(torch.nn.Module):
        def __init__(self):
            super(linearRegression, self).__init__()
            # input size: n_features, output size: 1
            self.linear = torch.nn.Linear(n_features, 1)
        def forward(self, x):
            out = self.linear(x)
            return out
    lr_model = linearRegression()

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(lr_model.parameters(), lr = 0.01)

    # Training loop
    for epoch in range(num_epochs):

        # Forward pass and loss
        pred_y = lr_model(X_train_tensor)
        loss = criterion(pred_y, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Make predictions on test set
    lr_model.eval()
    with torch.no_grad():
        y_pred_test = lr_model(X_test_tensor)

    end = time.time()

    y_pred = y_pred_test.numpy().flatten()
    y_true = y_test_tensor.numpy().flatten()

    # Compute RMSE
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    return rmse, end - start


def linear_regression_systemds(X_train, y_train, X_test, y_test, num_epochs=1000) -> tuple[float, float]:
    """
    Trains a linear regression model using SystemDS.

    :param X_train, X_test: numpy arrays of shape (n_samples, n_features)
    :param y_train, y_test: numpy arrays of shape (n_samples,)
    :param num_epochs: maximum number of training iterations

    :return rmse: RMSE on test set
    :return time taken: time in seconds for training and prediction
    """

    with SystemDSContext() as sds:

        start = time.time()

        # Read data into SystemDS
        X_ds = sds.from_numpy(X_train)
        y_ds = sds.from_numpy(y_train)
        X_test_ds = sds.from_numpy(X_test)

        # Train linear regression model with max iterations
        model = lm(X=X_ds, y=y_ds, maxi=num_epochs)
        # Make predictions on test set
        predictions = lmPredict(X_test_ds, model).compute()

        end = time.time()

        y_pred = predictions.flatten()
        y_true = y_test.flatten()

        # Compute RMSE
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    return rmse, end - start


def evaluate_lr() -> None:
    """
    Evaluates and compares the linear regression computations between PyTorch and SystemDS. The data is split into 
    training and test sets with an 80/20 ratio. Then the RMSE is calculated for both sets.
    """

    print("Evaluating Linear Regression Models...")

    X_train, y_train, X_test, y_test = preprocess_data()

    pytorch_rmse, pytorch_time = linear_regression_pytorch(X_train, y_train, X_test, y_test, num_epochs=1000)
    systemds_rmse, systemds_time = linear_regression_systemds(X_train, y_train, X_test, y_test, num_epochs=1000)

    print(f"PyTorch RMSE: {pytorch_rmse}, Time: {pytorch_time} seconds")
    print(f"SystemDS RMSE: {systemds_rmse}, Time: {systemds_time} seconds")


if __name__ == "__main__":

    # Cosine Similarity
    evaluate_cosine_similarity()
    
    # Matrix Factorization using ALS
    evaluate_als("systemds")
    evaluate_als("numpy")

    # Linear Regression
    evaluate_lr()
