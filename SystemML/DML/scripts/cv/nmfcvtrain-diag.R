NMFcvtrain = function(x,y,z) return (wb,hc) {
	max_iteration = 10
	accuracy = 0.000001
	numtopics = 4;
	
	zrow = nrow(z);
	wd = Rand(rows = zrow, cols = numtopics, rows_in_block=2, columns_in_block=2) ;
	zcol = ncol(z)
	hd = Rand(rows = numtopics, cols=zcol, rows_in_block=2, columns_in_block=2) ;
	continue = true
	i = 0
	while (continue && i < max_iteration) {
  		hd = hd * ((t(wd) %*% z) /  ( (t(wd) %*% wd) %*% hd))
  		wd = wd * ((z %*% t(hd)) / ( wd %*% (hd %*% t(hd))))
		i = i + 1
		
		# Convergence test begin
		oldObj = newObj
  		temp1 = (t(wd) %*% wd) * (hd %*% t(hd))
		temp2 = t(wd) %*% z %*% hd
		newObj = sum(z*z) + sum(diag(temp1)) - 2*sum(diag(temp2))
  		continue = (newObj-oldObj) > (oldObj * accuracy)
  		# Convergence test ends
	}
	
	xrow = nrow(x)
	numtopics = 4;
	wb = Rand(rows = xrow, cols = numtopics, rows_in_block=2, columns_in_block=2) ;
	continue = true
	i = 0
	while (continue && i < max_iteration) {
  		wb = wb * ((x %*% t(hd)) / ( wb %*% (hd %*% t(hd))))
  		i = i + 1
  		
  		# Convergence test begin
		oldObj = newObj
  		temp1 = (t(wb) %*% wb) * (hd %*% t(hd))
		temp2 = t(wb) %*% x %*% hd
		newObj = sum(x*x) + sum(diag(temp1)) - 2*sum(diag(temp2))
  		continue = (newObj-oldObj) > (oldObj * accuracy)
  		# Convergence test ends
	}
	
	ycol = ncol(y)
	numtopics = 4;
	hc = Rand(rows = numtopics, cols = ycol, rows_in_block=2, columns_in_block=2)
	continue = true
	i = 0
	while (continue && i < max_iteration) {
		hc = hc * ((t(wd) %*% y) / ( (t(wd) %*% wd) %*% hc))
		i = i + 1
		
		# Convergence test begin
		oldObj = newObj
  		temp1 = (t(wd) %*% wd) * (hc %*% t(hc))
		temp2 = t(wd) %*% y %*% hc
		newObj = sum(y*y) + sum(diag(temp1)) - 2*sum(diag(temp2))
		continue = (newObj-oldObj) > (oldObj * accuracy)
		# Convergence test ends
	}
}