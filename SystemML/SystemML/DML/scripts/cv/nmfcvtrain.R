NMFcvtrain = function(x,y,z) return (wb,hc) {
	zrow = nrow(z);
	numtopics = 4;
	wd = Rand(rows = zrow, cols = numtopics, rows_in_block=2, columns_in_block=2) ;
	
	zcol = ncol(z)
	numtopics = 4;
	hd = Rand(rows = numtopics, cols=zcol, rows_in_block=2, columns_in_block=2) ;
	
	#accuracy = 0.000001
	#continue = true
	#zerror = z - wd %*% hd
	#newObj = sum(derror * derror)
	
	max_iteration = 10
	i = 0
	#while (continue && i < max_iteration) {
	while (i < max_iteration) {
  		hd = hd * ((t(wd) %*% z) /  ( (t(wd) %*% wd) %*% hd))
  		wd = wd * ((z %*% t(hd)) / ( wd %*% (hd %*% t(hd))))
		i = i + 1
		
		# Convergence test begin
		#oldObj = newObj
  		#derror = z - wd %*% hd
  		#newObj = sum(derror * derror)
  		#continue = (newObj-oldObj) > (oldObj * accuracy)
	}
	
	xrow = nrow(x)
	numtopics = 4;
	wb = Rand(rows = xrow, cols = numtopics, rows_in_block=2, columns_in_block=2) ;
	#continue = true 
	#berror = x - wb %*% hd
	#newObj = sum(berror * berror)
	
	i = 0
	#while (continue && i < max_iteration) {
	while (i < max_iteration) {
  		wb = wb * ((x %*% t(hd)) / ( wb %*% (hd %*% t(hd))))
  		i = i + 1
  		
  		# Convergence test begin
		#oldObj = newObj
  		#berror = x - wb %*% hd
		#newObj = sum(berror * berror)
  		#continue = (newObj-oldObj) > (oldObj * accuracy)
	}
	
	ycol = ncol(y)
	numtopics = 4;
	hc = Rand(rows = numtopics, cols = ycol, rows_in_block=2, columns_in_block=2)
	#continue = true
	#cerror = y - wd %*% hc
	#newObj = sum(cerror * cerror)
	
	i = 0
	#while (continue && i < max_iteration) {
	while (i < max_iteration) {
		hc = hc * ((t(wd) %*% y) /  ( (t(wd) %*% wd) %*% hc))
		i = i + 1
		
		# Convergence test begin
		#oldObj = newObj
  		#cerror = y - wd %*% hc
		#newObj = sum(cerror * cerror)
		#continue = (newObj-oldObj) > (oldObj * accuracy)
	}
}