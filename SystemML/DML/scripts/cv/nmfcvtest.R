NMFcvtest = function (alpha,beta,gamma) return (error)
{
	apred = beta %*% gamma
	diff = alpha - apred 
	product = diff*diff
	error = rowSums(colSums(product))
}