mydata = readMM ("matrix", rows=10, cols = 10, nnzs=20, format="text") ;

create_ensemble myensemble
data mydata
partition(method='bootstrap', element='row', numIterations=3) as dataset
train trainFunction(dataset) as model

using myensemble
test testFunction(mytestdata, model) as output
AGG avg;