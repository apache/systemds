NMFcvtrain = function(x,y,z) return (wb,hc) {
	wd=readMM("example.GNMF.wd", rows=5, cols=5, nnzs=20, format="text") ;
	# wd = rand(rows=numrows(z), cols = 10) ;
	hd=readMM("example.GNMF.hd", rows=5, cols=5, nnzs=20, format="text") ;
	# hd = rand(rows = 10, cols=numcolumns(z)) ;
	
	accuracy = 0.000001
	
	max_iteration = 10
	i = 0
	continue = true
	derror = d - wd %*% hd
  	newObj = sum(derror * derror)
	
	while (continue && i < max_iteration) {
  		hd = hd * ((t(wd) %*% z) /  ( (t(wd) %*% wd) %*% hd))
  		wd = wd * ((z %*% t(hd)) / ( wd %*% (hd %*% t(hd))))
		
		# Convergence test begin
		oldObj = newObj
  		derror = d - wd %*% hd
  		newObj = sum(derror * derror)
  		if ((newObj-oldObj) < oldObj * accuracy)
  			continue = false ;
		# Convergence test end
		
  		i = i + 1
	}
	
	wb=readMM("example.GNMF.wb", rows=5, cols=5, nnzs=20, format="text") ;
	# wb = rand(rows = numrows(x), cols = 10) ;
	
	i = 0
	continue = true 
	berror = x - wb %*% hb
	newObj = sum(berror * berror)
	
	while (continue && i < max_iteration) {
  		wb = wb * ((x %*% t(hd)) / ( wb %*% (hd %*% t(hd))))
  		
  		# Convergence test begin
		oldObj = newObj
  		berror = x - wb %*% hd
		newObj = sum(berror * berror)
  		if ((newObj-oldObj) < oldObj * accuracy)
  			continue = false ;
		# Convergence test end
  		
  		i = i + 1
	}
	
	hc=readMM("example.GNMF.hc", rows=5, cols=5, nnzs=20, format="text") ;
	# hc = rand(rows = 10, cols = numcolumns(y))
	i = 0
	continue = true
	cerror = y - wd %*% hc
	newObj = sum(cerror * cerror)
	
	while (continue && i < max_iteration) {
		hc = hc * ((t(wd) %*% y) /  ( (t(wd) %*% wd) %*% hc))
		
		# Convergence test begin
		oldObj = newObj
  		cerror = y - wd %*% hc
		newObj = sum(cerror * cerror)
  		if ((newObj-oldObj) < oldObj * accuracy)
  			continue = false ;
		# Convergence test end
		
  		i = i + 1
	}
}