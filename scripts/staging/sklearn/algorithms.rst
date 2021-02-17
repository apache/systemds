Supported Algorithms
====================

Classification
--------------

Supervised
""""""""""
* glm.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html#sklearn.linear_model.TweedieRegressor
* l2svm.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
* lm.dml/lmCG.dml/lmDS.dml (lmpredict.dml) <-> https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression
* multiLogReg.dml (multiLogRegPredict.dml) <-> https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Unsupervised
""""""""""""
* dbscan.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
* kmeans.dml (kmeansPredict.dml) <-> https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
* gmm.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture

Transformations
---------------
* split.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
* scale.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
* normalize.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer
* toOneHot.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
* pca.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA
* imputeByMean.dml/imputeByMedian.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer

Scoring
-------
* dist.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html#sklearn.metrics.pairwise.euclidean_distances
* getAccuracy.dml <-> https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score


Maybe
-----
* confusionMatrix.dml
* cvlm.dml
* imputeByFD.dml
* lasso.dml
* msvm.dml
* msvmPredict.dml
* na_locf.dml
* steplm.dml
* naivebayes.dml <-> https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes