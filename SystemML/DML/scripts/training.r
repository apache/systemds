x = readMM ( "xblocked",format="binary",value_type="double",cols=5,columns_in_block=2,rows_in_block=2,rows=5);
y = readMM ( "yblocked",format="binary",value_type="double",cols=5,columns_in_block=2,rows_in_block=2,rows=5);
z = readMM ( "zblocked",format="binary",value_type="double",cols=5,columns_in_block=2,rows_in_block=2,rows=5);
wd = Rand(cols=5,rows=5,columns_in_block=2,rows_in_block=2) ;
hd = Rand(cols=5,rows=5,columns_in_block=2,rows_in_block=2) ;

#readMM ( "example.GNMF.wd",format="binary",cols=5,columns_in_block=2,rows_in_block=2,nnzs=20,rows=5);
#hd = readMM ( "example.GNMF.hd",format="binary",cols=5,columns_in_block=2,rows_in_block=2,nnzs=20,rows=5);

#xrow = nrow(x)

max_iteration = 1
i = 0
while (i < max_iteration) {
 	hd = hd * ((t(wd) %*% z) /  ( (t(wd) %*% wd) %*% hd))
  	wd = wd * ((z %*% t(hd)) / ( wd %*% (hd %*% t(hd))))
	i = i + 1
}

#writeMM(wd,"wd.out",format="binary")
#writeMM(hd,"hc.out",format="binary")

	
wb=readMM("example.GNMF.wd", rows=5, cols=5, nnzs=20, rows_in_block=2, columns_in_block=2, format="binary") ;
# wb = rand(rows = numrows(x), cols = 10) ;
i = 0
while (i < max_iteration) {
 	wb = wb * ((x %*% t(hd)) / ( wb %*% (hd %*% t(hd))))
  	i = i + 1
}

#writeMM(wb,"wb.out",format="binary")
#writeMM(wd,"wd.out",format="binary")
#writeMM(hd,"hd.out",format="binary")
	
hc=readMM("example.GNMF.hd", rows=5, cols=5, nnzs=20, rows_in_block=2, columns_in_block=2, format="binary") ;
# hc = rand(rows = 10, cols = numcolumns(y))
i = 0
while (i < max_iteration) {
	hc = hc * ((t(wd) %*% y) /  ( (t(wd) %*% wd) %*% hc))
	i = i + 1
}

writeMM(wb,"wb.out",format="binary")
writeMM(hc,"hc.out",format="binary")