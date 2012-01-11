 
	x=readMM("example.GNMF.x", rows=5, cols=5, nnzs=20, format="text") ;
	y=readMM("example.GNMF.y", rows=5, cols=5, nnzs=20, format="text") ;
	z=readMM("example.GNMF.z", rows=5, cols=5, nnzs=20, format="text") ;
	
	zrow = nrow(z);
	numtopics = 4;
	wd = Rand(rows = zrow, cols = numtopics) ;
	
	zcol = ncol(z)
	#numtopics = 4;
	hd = Rand(rows = numtopics, cols=zcol) ;
	
	accuracy = 0.000001
	continue = true
	zerror = z - wd %*% hd
	newObj = sum(zerror * zerror)
	
	max_iteration = 10
	i = 0
	while (continue == true & i < max_iteration) {
	#while (i < max_iteration) {
  		hd = hd * ((t(wd) %*% z) /  ( (t(wd) %*% wd) %*% hd))
  		wd = wd * ((z %*% t(hd)) / ( wd %*% (hd %*% t(hd))))
		i = i + 1
		
		# Convergence test begin
		oldObj = newObj
  		derror = z - wd %*% hd
  		newObj = sum(derror * derror)
  		continue = (newObj-oldObj) > (oldObj * accuracy)
	}
	
	xrow = nrow(x)
	#numtopics = 4;
	wb = Rand(rows = xrow, cols = numtopics) ;
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
	#numtopics = 4;
	hc = Rand(rows = numtopics, cols = ycol)
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
	writeMM (wb,"dml/scripts/nmf.wb.result", rows_in_block = 1, columns_in_block = 1, format="binary");
	writeMM (hc,"dml/scripts/nmf.hc.result", rows_in_block = 1, columns_in_block = 1, format="binary");
	