import numpy as np
from sklearn.datasets import make_blobs

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from map_pipeline import SklearnToDMLMapper

X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=1)

np.savetxt('X.csv', X, delimiter=",")

pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters = 3, random_state=1))

model = pipeline.fit(X)

print(model)

print('Labels:', model.steps[1][1].labels_)
#print('Centers:', model.steps[1][1].cluster_centers_)

mapper = SklearnToDMLMapper(pipeline)

mapper.transform()
mapper.save('pipeline.dml')