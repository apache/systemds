# Overall syntax
[<train_name>,<test_name>] =  partition(
									<data>, 
          							<'kfold'|'holdout'|'boostrap'>, 
          							<'row'|'cell'|'submatrix'>, 
          							<k | frac | h 'X' l>
          							[, <idToStratify>, <domainSizeOfId>]
         					  )

partition <data>
using method <'kfold'|'holdout'|'bootstrap'>
element <'row','cell','submatrix'>
params <k | frac | h 'X' l>
[stratify on <idToStratify> domainSize <domainSizeOfId>]

# For 3-fold, row-wise, stratified (binary) partitioning of X
[train,test] = partition(X, 'kfold', 'row', 3, id, 2)
partition X using method 'kfold' element 'row' params 3 stratify on id domainSize 2
# Output will be 6 files ("test0", "test1", "test2", "train0", "train1", "train2")

# For 3-fold, row-wise partitioning (no stratification)
partition X using method 'kfold' element 'row' params 3
[train,test] = partition(X, 'kfold', 'row', 3)

# For holdout row-wise partitioning (no stratification)
[train,test] = partition(X, 'holdout', 'row', 0.4)
partition X using method 'holdout' element 'row' params 0.4
# Output will be 2 files ("test0", "train0"). test0 is 40% of dataset, train0 is 60% of X

# For NMF crossvalidation (slightly different)
[S] = partition(X, 'holdout', 'submatrix', 2X2)
partition X using method 'holdout' element 'submatrix' params 2X2
# Output will the S matrix (In reality the complement of the S matrix)

# For bootstrap sampling (Ensemble Learning)
[output] = partition(X, 'bootstrap', 'row', 4)
partition X using method 'bootstrap' element 'row' params 4
# Output will be 4 files ("output0", "output1", "output2", "output3")