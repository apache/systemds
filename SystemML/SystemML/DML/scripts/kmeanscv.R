# Please note that this file has not yet been tested.

partition X using method 'kfold' element 'row' params 3 outputdescriptor "train","test"

crossvalidate X
using method 'kfold'
element 'row'
params 3
[iterations 1]
outputdescriptor "train","test"
with cvtrain model = kmeanscvtrain(train)
with cvtest kmeanscvtest(test,model)


kmeanscvtrain = function(d) {
	c = kmeansclustering(d,5) #centroids
	return (c=c)
}

# Very simple cvtest
kmeanscvtest = function(dt,model) {
	distances = distmatrix(dt,c)
	numRows = nrow(dt)
	error = 0
	for (i in numRows) {
		error = error + min.distances[i,]
	}
	return (error=error)
}

kmeansclustering = function(d,k) {
	hA = nrow(d)
	wA = ncol(d)
	if(hA <= k) {
		return (output = d)
	}
	# Initialize the k centroids

	c = numeric() #Centroids
	P = as.integer(runif(k,1,hA))
	for(i in 1:k) {
		c = rbind(c,d[p[i],])
	}

	cids = numeric() #ClusterIds
	while(TRUE) {
		distances = distmatrix(d,c)
		sumDistances = matrix(0,k,1)
		counters = matrix(0,k,1) 
		for(i in 1:hA) {
			minDistance = min.distances[i,]
			tempVar = which.min(distances[i,])
			cids = rbind(cids,tempVar)
			sumDistances[tempVar,] = sumDistances[tempVar,] + minDistance
			counter[tempVar,] = counter[tempVar,] + 1
		}
		for(i in 1:k)
			c[i] = sumDistances[i,] / counter[i,]
	}
	return (c=c)
}

distmatrix = function(A,B) {
	hA = nrow(A)
	wA = ncol(A)
	hB = nrow(B)
	wB = ncol(B)

	if (wA != wB)
		print("Wrong dimensions")

	S=matrix(0,hA,hB);
	for(k in 1:wA) {
		C <- A[,rep(k:k,hB)]
		D <- B[,rep(k:k,hA)]
		S=S+(C-t(D))^2;
	}
      d=sqrt(S);
	return (d = d)
}