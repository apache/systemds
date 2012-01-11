# Purpose: 1st example to get going on statements blocks (then while loops, then if-then-else)
#    Functions: 
 
V = readMM("matrix",rows=10,cols=10,nnzs=20,format="text")
W = readMM ("mydataWblocked", rows= 10, cols=5, nnzs= 20, format="binary",rows_in_block=2, columns_in_block=2);
H = readMM ("mydataHblocked", rows= 5, cols=10, nnzs= 20, format="binary",rows_in_block=2, columns_in_block=2);

max_iteration = 10
i = 0

while (i < max_iteration) {
  H = H * ((t(W) %*% V) /  ( (t(W) %*% W) %*% H))
  W = W * ((V %*% t(H)) / ( W %*% (H %*% t(H))))
  i = i + 1  
}
 
#writeMM (W,"example.GNMF.W.result", format="text");
writeMM (W,"example.GNMF.W.result", format="binary");
#writeMM (H,"example.GNMF.H.result", format="text");
writeMM (H,"example.GNMF.H.result", format="binary");

