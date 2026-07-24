.. -------------------------------------------------------------
.. 
.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
.. 
..   http://www.apache.org/licenses/LICENSE-2.0
.. 
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.
.. 
.. ------------------------------------------------------------


Building a Movie Recommender System
===============

Have you ever wondered how Netflix, Disney+, and other streaming
platforms know exactly which movies or TV shows to recommend to you? In
this tutorial, we will explore how recommender systems work and show how
to implement them using SystemDS, as well as NumPy and PyTorch for
comparison. The goal of this tutorial is to showcase different features
of the SystemDS framework that can be accessed with the Python API.

In this tutorial, we will explore the implementation of a recommender
system using three distinct mathematical and machine learning
approaches:

-  **Cosine Similarity**: A geometric approach to measure the similarity
   between users or items based on the angle between their preference
   vectors.

-  **Matrix Factorization**: A technique often used in latent factor
   models (like ALS) to decompose the user-item interaction matrix into
   lower-dimensional representations.

-  **Linear Regression**: A supervised learning approach used to predict
   specific ratings by modeling the relationship between user/item
   features and the target rating.

This tutorial shows only snippets of the code and the whole code can be
found
`here <https://github.com/apache/systemds/tree/main/src/main/python/systemds/examples/tutorials/movie_recommender_system.py>`__.

To start with the tutorial, you first have to install SystemDS: :doc:`/getting_started/install`.


Dataset
~~~~~~~

As a dataset we chose the `MovieLens 100K
Dataset <https://grouplens.org/datasets/movielens/100k/>`__. It consists
of 100.000 movie ratings from 943 different users on 1682 movies from
the late 1990s. In the following, we will often refer to movies as
items. The data is stored in different files: 

-  **u.data** (contains user_id, item_id (movie), rating and timestamp), 

-  **u.user** (contains user_id, age, gender, occupation and zip code), 

-  **u.item** (contains item_id, movie name, release date, ImDb link, genre in a hot-one format).

Preprocessing for Cosine Similarity and Matrix Factorization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To prepare our data for Cosine Similarity and Matrix Factorization, we
must convert the raw ratings into a User-Item Interaction Matrix. In
this structure, each row represents a unique user, and each column
represents a specific movie. The intersection of a row and column
contains the user’s rating for that movie.

Because the average user has only seen a small percentage of the
thousands of movies available, this matrix is extremely sparse. Most
cells will contain missing values (NaN), which we must handle
differently depending on the algorithm we choose.

First, we load the MovieLens 100k dataset:

.. code:: python

   # Load the MovieLens 100k dataset
   header = ['user_id', 'item_id', 'rating', 'timestamp']
   ratings_df = pd.read_csv('movie_data/ml-100k/u.data', sep='\t', names=header)

We then use the Pandas ``.pivot()`` function to transform the data. This
gives us the User-Item table.

.. code:: python

   pivot_df = ratings_df.pivot(index='user_id', columns='item_id', values='rating')

The resulting matrix provides a high-level view of our dataset’s
interaction patterns:

====== ====== ====== ======
User   Item 1 Item 2 Item 3
====== ====== ====== ======
user 1 5             3
user 2        4      2
user 3 1      5      
====== ====== ====== ======

Cosine Similarity
~~~~~~~~~~~~~~~~~

Collaborative Filtering is an umbrella term for algorithms that generate
recommendations by identifying patterns in user ratings. One of the most
common techniques is User-Based Collaborative Filtering, where we
calculate the similarity between users based on their rating history. To
do this, we treat each user as a vector in a high-dimensional space and
measure the “distance” between them using `Cosine Similarity <https://towardsdatascience.com/cosine-similarity-how-does-it-measure-the-similarity-maths-behind-and-usage-in-python-50ad30aad7db/>`__.

To calculate the cosine similarity between all users (rows) in a matrix
:math:`X`, we normalize this matrix and then multiply it with its
transposose.

If :math:`\hat{X}` is the row-normalized version of :math:`X` such that
each row :math:`i` is defined as
:math:`\hat{x}_i = \frac{x_i}{\|x_i\|}`, then the entire Cosine
Similarity matrix :math:`S` is calculated via the gramian matrix:

.. math:: S = \hat{X}\hat{X}^T

Using NumPy, we perform these operations using vectorization for
efficiency:

.. code:: python

   # L2 norm of each user vector
   norms = np.linalg.norm(X, axis=1, keepdims=True)

   # Normalize user vectors
   X_norm = X / norms

   # Cosine similarity = dot product of normalized vectors
   user_similarity = X_norm @ X_norm.T

In SystemDS, we follow a similar logic. First, we import and initialize
the SystemDSContext.

.. code:: python

   from systemds.context import SystemDSContext

   with SystemDSContext() as sds:

In this context window, we load the data into SystemDS and do our
calculations, using simple matrix
functions: :doc:`/api/operator/node/matrix`.

.. code:: python

   # Load into SystemDS
   X = sds.from_numpy(X_np)

   # Compute L2 norms 
   row_sums = (X * X).sum(axis=1)
   norms = row_sums.sqrt()

   # Normalize user vectors
   X_norm = X / norms

   # Cosine similarity = dot product of normalized vectors
   user_similarity = X_norm @ X_norm.t()

In SystemDS, the line ``user_similarity_op = X_norm @ X_norm.t()`` does
not execute any math. Instead, it creates an execution plan. The actual
computation only occurs when we call ``.compute()``, allowing SystemDS
to optimize the entire operation.

.. code:: python

   user_similarity = user_similarity.compute()

In both cases ``user_similarity`` gives us a diagonal matrix that shows the
similarity for every user-user pair.

While both methods produce the same results, SystemDS takes slightly
longer for this specific dataset.

=============== ===== ========
Method          NumPy SystemDS
=============== ===== ========
Time in seconds 0.02  0.47
=============== ===== ========

Matrix Factorization
~~~~~~~~~~~~~~~~~~~~

Another powerful method for generating movie recommendations is Matrix
Factorization. Instead of looking at surface-level data, this technique
uncovers latent factors, the hidden patterns that represent a user’s
specific tastes (like a preference for 90s rom-coms) and a movie’s
unique characteristics (like its level of whimsy).

In a real-world scenario, our user-item interaction matrix :math:`R`
is incredibly sparse because most users have only rated a tiny fraction
of the available movies. Matrix factorization solves this by decomposing
:math:`R` into two much smaller, lower-dimensional matrices:

-  :math:`P`: Representing user preferences.
-  :math:`Q`: Representing item characteristics.

By multiplying these two matrices back together, we can estimate the
missing values in our original matrix:

.. math::  R \approx P \cdot Q^T

To find :math:`P` and :math:`Q`, we use the optimization algorithm
called 
`Alternating Least Squares (ALS) <https://www.shaped.ai/blog/matrix-factorization-the-bedrock-of-collaborative-filtering-recommendations>`__.

In NumPy, we manually iterate through users and items, solving a
least-squares problem for each. This gives us full control but can be
computationally expensive as the dataset grows. We can compute
:math:`\hat{R} = P \cdot Q^T` like this
`(cf. CodeSignal) <https://codesignal.com/learn/courses/diving-deep-into-collaborative-filtering-techniques-with-als/lessons/implementing-the-alternating-least-squares-algorithm>`__:

.. code:: python

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

   R_hat = P @ Q.T

SystemDS allows us to execute the same logic using high-level
script-like functions that are internally optimized. It offers a wide
variety of built-in algorithms  :doc:`/api/operator/algorithms`, including ALS.
First, we import our algorithm.

.. code:: python

   from systemds.operator.algorithm import als

Then, we initialize the SystemDS context:

.. code:: python

   with SystemDSContext() as sds:

To tune the model for our specific dataset, we configure the following
hyperparameters:

-  ``rank = 20`` The number of latent factors (hidden features) used to
   describe users and movies. A higher rank allows for more complexity
   but increases the risk of overfitting.
-  ``reg = 1.0`` The regularization parameter. This prevents the model
   from becoming too complex by penalizing large weights, helping it
   generalize better to unseen data.
-  ``maxi = 20`` The maximum number of iterations. ALS is an iterative
   process.

Then we can do the computation.

.. code:: python

   # Load data into SystemDS
   R = sds.from_numpy(pivot_df.fillna(0).values)

   # Approximate factorization of R into two matrices P and Q using ALS
   P, Q = als(R, rank=20, reg=1.0, maxi=20).compute()

   R_hat = P @ Q

To test how well our models generalize to new data, we performed an
80/20 split, using the first 80,000 ratings for training and the
remainder for testing. We compared both approaches based on execution
time and Root Mean Squared Error (RMSE).

=============== ===== ========
Method          NumPy SystemDS
=============== ===== ========
Time in seconds 1.09  2.48
Train RMSE      0.67  0.67
Test RMSE       1.03  1.01
=============== ===== ========

Both implementations are mathematically consistent. SystemDS achieved a
slightly better Test RMSE.

Linear Regression
~~~~~~~~~~~~~~~~~

Unlike Matrix Factorization, which relies purely on interaction
patterns, Linear Regression allows us to incorporate “side information”
about users and items. By using features like user demographics and
movie genres, we can build a Content-Based Filtering model that predicts
ratings based on specific attributes.

Preprocessing
^^^^^^^^^^^^^

For Linear Regression and Neural Networks, our data must be strictly
numerical and properly scaled. We begin by loading the MovieLens
datasets:

.. code:: python

   ratings_df = pd.read_csv('movie_data/ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
   user_df = pd.read_csv('movie_data/ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
   item_df = pd.read_csv('movie_data/ml-100k/u.item', sep='|', names=[
       'item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
       "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')

Libraries like NumPy and SystemDS cannot process strings (e.g.,
“Student” or “Female”). We must transform these into numerical
representations:

.. code:: python

   # Turn categorical data into numerical
   user_df['gender'] = user_df['gender'].apply(lambda x: 0 if x == 'F' else 1)
   user_df = pd.get_dummies(user_df, columns=['occupation'])
   item_df['release_date'] = pd.to_datetime(item_df['release_date'], errors='raise', format='%d-%b-%Y')
   item_df['release_year'] = item_df['release_date'].dt.year

Features like ``age`` and ``release_year`` have different scales. If
left unscaled, the model might incorrectly give more “weight” to the
larger year values. We normalize them to a 0–1 range to ensure equal
influence.

.. code:: python

   # Normalize data
   user_df['age'] = (user_df['age'] - user_df['age'].min()) / (user_df['age'].max() - user_df['age'].min())
   item_df['release_year'] = (item_df['release_year'] - item_df['release_year'].min()) / (item_df['release_year'].max() - item_df['release_year'].min())

Finally, we merge these datasets into a single table. Each row
represents a specific rating, enriched with all available user and movie
features. After merging, we drop non-numerical columns (like ``title``
or ``IMDb_URL``), remove rows with NaN-values and split the data into
Training and Testing sets.

Linear Regression with PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch is a popular deep learning framework that approaches linear
regression as an iterative optimization problem. We use Gradient Descent
to minimize the Mean Squared Error (MSE) by repeatedly updating the
model’s weights based on calculated gradients.

Data must be converted to ``torch.Tensor`` format.

.. code:: python

   X_train_tensor = torch.from_numpy(X_train).float()
   y_train_tensor = torch.from_numpy(y_train).float().reshape(-1, 1)
   X_test_tensor = torch.from_numpy(X_test).float()
   y_test_tensor = torch.from_numpy(y_test).float().reshape(-1, 1)

We define a model class and an optimizer (SGD). The learning rate
(``lr``) determines the step size for each update.

.. code:: python

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
   criterion = torch.nn.MSELoss()
   optimizer = torch.optim.SGD(lr_model.parameters(), lr = 0.01)

The model iterates through the dataset for a set number of epochs. In
each iteration, it performs a forward pass, calculates the loss, and
backpropagates the gradients to update the weights.

.. code:: python

   for epoch in range(1000):

       # Forward pass and loss
       pred_y = lr_model(X_train_tensor)
       loss = criterion(pred_y, y_train_tensor)

       # Backward pass and optimization
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

We use ``.eval()`` and ``torch.no_grad()`` to disable gradient tracking
during inference

.. code:: python

   lr_model.eval()
   with torch.no_grad():
       y_pred_test = lr_model(X_test_tensor)

Then, we can calculate the RMSE.

.. code:: python

    y_pred = y_pred_test.numpy().flatten()
    y_true = y_test_tensor.numpy().flatten()
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

Linear Regression with SystemDS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following the same pattern as ALS, SystemDS
provides a highly optimized, built-in algorithm for linear regression.
This implementation is designed to handle large-scale data by
automatically deciding between direct solvers and conjugate gradient
methods based on the data’s characteristics.

First, we import the ``lm`` training algorithm and the ``lmPredict``
function for inference.

.. code:: python

   from systemds.operator.algorithm import lm, lmPredict

We transfer our NumPy arrays into the SystemDS context.

.. code:: python

   X_ds = sds.from_numpy(X_train)
   y_ds = sds.from_numpy(y_train)
   X_test_ds = sds.from_numpy(X_test)

We call the ``lm`` function to train our model.

.. code:: python

   model = lm(X=X_ds, y=y_ds)

To generate predictions for the test set, we use ``lmPredict``. Because
SystemDS uses Lazy Evaluation, the actual computation is only triggered
when we call ``.compute()``.

.. code:: python

   predictions = lmPredict(X_test_ds, model).compute()

Finally, we calculate the RMSE to compare the performance against your
PyTorch implementation.

Comparison
^^^^^^^^^^

=============== ======= ========
Method          PyTorch SystemDS
=============== ======= ========
Time in seconds 1.77    0.87
Test RMSE       1.13    1.08
=============== ======= ========

Using linear regression, SystemDS worked way faster than our PyTorch
approach and achieved better results.
