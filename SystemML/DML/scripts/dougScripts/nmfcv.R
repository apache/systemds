mydata = readMM ("matrix", rows=10, cols = 10, nnzs=30, format="text") ;

crossval mydata
partition (method='kfold', element='submatrix', numRowGroups=2, numColGroups=2, rows_in_block=1000, columns_in_block=1000) as (a,b,c,d) 
train nmfcvtrain1(d) as (w,h)
#train nmfcvtest2(a) as (error)
test nmfcvtest1(b,c,w,h,a) as (error)
aggregate sum(error) as (err)