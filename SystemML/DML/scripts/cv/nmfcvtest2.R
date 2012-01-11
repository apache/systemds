NMFcvtest = function (alpha) return (error)
{
	beta = readMM("w",rows=5,cols=5,format="binary",rows_in_block=2,columns_in_block=2) ;
	gamma = readMM("h", rows=5,cols=5,format="binary",rows_in_block=2,columns_in_block=2) ;
	apred = beta %*% gamma
	diff = alpha - apred 
	product = diff * diff
	#row_error = rowSums(product)
	#error = colSums(row_error)
	error = sum(product)
}