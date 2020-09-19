#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

args<-commandArgs(TRUE)
options(digits=22)
library("Matrix")
library("matrixStats")

# library("doMC")
# registerDoMC(NULL) # physical cores

################################################################################

slicefinder = function(X, e,
  k = 4, maxL = 0, minSup = 32, alpha = 0.5,
  tpEval = TRUE, tpBlksz = 16, verbose = FALSE)
{
  # init debug matrix: levelID, enumerated S, valid S, TKmax, TKmin
  D = matrix(0, 0, 5);
  m = nrow(X);
  n = ncol(X);
  
  # prepare offset vectors and one-hot encoded X
  fdom = colMaxs(X);
  foffb = t(cumsum(t(fdom))) - fdom;
  foffe = t(cumsum(t(fdom)))
  rix = matrix(seq(1,m)%*%matrix(1,1,n), m*n, 1)
  cix = matrix(X + (matrix(1,nrow(X),1) %*% foffb), m*n, 1);
  X2 = table(rix, cix); #one-hot encoded

  # initialize statistics and basic slices
  n2 = ncol(X2);     # one-hot encoded features
  eAvg = sum(e) / m; # average error
  TMP1 = createAndScoreBasicSlices(X2, e, eAvg, minSup, alpha, verbose); 
  S = TMP1[["S"]];
  R = TMP1[["R"]];

  # initialize top-k
  TMP2 = maintainTopK(S, R, matrix(0,0,n2), matrix(0,0,4), k, minSup);
  TK = TMP2[["TK"]]; TKC = TMP2[["TKC"]];

  if( verbose ) {
    TMP3 = analyzeTopK(TKC);
    maxsc = TMP3[["maxsc"]]; minsc = TMP3[["minsc"]];
    print(paste("SliceFinder: initial top-K: count=",nrow(TK),", max=",maxsc,", min=",minsc,sep=""))
    D = rbind(D, t(as.matrix(list(1, n2, nrow(S), maxsc, minsc))));
  }

  # lattice enumeration w/ size/error pruning, one iteration per level
  # termination condition (max #feature levels)
  maxL = ifelse(maxL<=0, n, maxL)
  level = 1;
  while( nrow(S) > 0 & sum(S) > 0 & level < n & level < maxL ) {
    level = level + 1;

    # enumerate candidate join pairs, incl size/error pruning 
    nrS = nrow(S);
    S = getPairedCandidates(S, R, TK, TKC, k, level, eAvg, minSup, alpha, n2, foffb, foffe); 

    if(verbose) {
      print(paste("SliceFinder: level ",level,":",sep=""))
      print(paste(" -- generated paired slice candidates: ",nrS," -> ",nrow(S),sep=""));
    }

    # extract and evaluate candidate slices
    if( tpEval ) { # task-parallel
      #R = foreach( i=1:nrow(S), .combine=rbind) %dopar% {
      #  return (evalSlice(X2, e, eAvg, as.matrix(S[i,]), level, alpha))
      #}
      R = matrix(0, nrow(S), 4)
      for(i in 1:nrow(S)) {
        R[i,] = evalSlice(X2, e, eAvg, as.matrix(S[i,]), level, alpha)
      }
    }
    else { # data-parallel
      R = evalSlice(X2, e, eAvg, t(S), level, alpha);
    }
    
    # maintain top-k after evaluation
    TMP2 = maintainTopK(S, R, TK, TKC, k, minSup);
    TK = TMP2[["TK"]]; TKC = TMP2[["TKC"]];

    if(verbose) {
      TMP3 = analyzeTopK(TKC);
      maxsc = TMP3[["maxsc"]]; minsc = TMP3[["minsc"]];
      valid = as.integer(sum(R[,2]>0 & R[,4]>=minSup));
      print(paste(" -- valid slices after eval: ",valid,"/",nrow(S),sep=""));
      print(paste(" -- top-K: count=",nrow(TK),", max=",maxsc,", min=",minsc,sep=""));
      D = rbind(D, t(as.matrix(list(level, nrow(S), valid, maxsc, minsc))));
    }
  }

  TK = decodeTopK(TK, foffb, foffe);

  if( verbose ) {
    print(paste("SliceFinder: terminated at level ",level,":"));
    print(TK); print(TKC);
  }
  
  return (list("TK"=TK, "TKC"=TKC, "D"=D))
}

rexpand = function(v, n2=max(v)) {
  R = matrix(0, nrow(v), n2)
  for( i in 1:nrow(v) ) {
    R[i,] = tabulate(v[i,], nbins=n2);
  }
  return (R)
} 

createAndScoreBasicSlices = function(X2, e, eAvg, minSup, alpha, verbose) {
  n2 = ncol(X2);
  cCnts = colSums(X2);    # column counts
  err = t(t(e) %*% X2);   # total error vector
  merr = t(colMaxs(X2 * (e %*% matrix(1,1,ncol(X2))))); # maximum error vector

  if( verbose ) {
    drop = as.integer(sum(cCnts < minSup | err == 0));
    print(paste("SliceFinder: dropping ",drop,"/",n2," features below minSup = ",minSup,".", sep=""));
  }

  # working set of active slices (#attr x #slices) and top k
  selCols = (cCnts >= minSup & err > 0);
  attr = as.matrix(seq(1,n2)[selCols])
  ss = as.matrix(cCnts[selCols])
  se = as.matrix(err[selCols])
  sm = as.matrix(merr[selCols])
  S = rexpand(attr, n2);
  
  # score 1-slices and create initial top-k 
  sc = score(ss, se, eAvg, alpha, nrow(X2));
  R = as.matrix(cbind(sc, se, sm, ss));

  return (list("S"=S,"R"=R))
}

score = function(ss, se, eAvg, alpha, n) {
  sc = alpha * ((se/ss) / eAvg - 1) - (1-alpha) * (n/ss - 1);
  sc = replace(sc, NaN, -Inf);
  return (sc)
}

scoreUB = function(ss, se, sm, eAvg, minSup, alpha, n) {
  # Since sc is either monotonically increasing or decreasing, we
  # probe interesting points of sc in the interval [minSup, ss],
  # and compute the maximum to serve as the upper bound 
  s = cbind(matrix(minSup,nrow(ss),1), pmax(se/sm,minSup), ss)
  ex = matrix(1,1,3)
  sc = rowMaxs(alpha * ((pmin(s*(sm%*%ex),se%*%ex)/s) / eAvg - 1) - (1-alpha) * (1/s*n - 1));
  sc = replace(sc, NaN, -Inf);
  return (sc)
}


maintainTopK = function(S, R, TK, TKC, k, minSup) {
  # prune invalid minSup and scores
  I = as.matrix((R[,1] > 0) & (R[,4] >= minSup));

  if( sum(I)!=0 ) {
    S = as.matrix(S[I,])
    R = as.matrix(R[I,])
    if( ncol(S) != ncol(TK) & ncol(S)==1 ) {
      S = t(S); R = t(R);
    }

    # evaluated candidated and previous top-k
    slices = as.matrix(rbind(TK, S));
    scores = as.matrix(rbind(TKC, R));

    # extract top-k
    IX = as.matrix(order(scores[,1], decreasing=TRUE));
    IX = as.matrix(IX[1:min(k,nrow(IX)),]);
    TK = as.matrix(slices[IX,]);
    TKC = as.matrix(scores[IX,]);
  }
  return (list("TK"=TK, "TKC"=TKC))
}

analyzeTopK = function(TKC) {
  maxsc = -Inf;
  minsc = -Inf;
  if( nrow(TKC)>0 ) {
    maxsc = TKC[1,1];
    minsc = TKC[nrow(TKC),1];
  }
  return (list("maxsc"=maxsc, "minsc"=minsc))
}

getPairedCandidates = function(S, R, TK, TKC, k,
  level, eAvg, minSup, alpha, n2, foffb, foffe)
{
  # prune invalid slices (possible without affecting overall
  # pruning effectiveness due to handling of missing parents)
  pI = (R[,4] >= minSup & R[,2] > 0);
  S = S[pI,]; R = R[pI,];

  # join compatible slices (without self)
  join = S %*% t(S) == (level-2)
  I = upper.tri(join, diag=FALSE) * join;

  # pair construction
  nr = nrow(I); nc = ncol(I);
  rix = matrix(I * (seq(1,nr)%*%matrix(1,1,ncol(I))), nr*nc, 1);
  cix = matrix(I * (matrix(1,nrow(I),1) %*% t(seq(1,nc))), nr*nc, 1);
  rix = as.matrix(rix[rix!=0,])
  cix = as.matrix(cix[cix!=0,])
  
  P = matrix(0, 0, ncol(S))
  if( sum(rix)!=0 ) {
    P1 = rexpand(rix, nrow(S));
    P2 = rexpand(cix, nrow(S));
    P12 = P1 + P2; # combined slice
    P = ((P1 %*% S + P2 %*% S) != 0) * 1;
    se = pmin(P1 %*% R[,2], P2 %*% R[,2])
    sm = pmin(P1 %*% R[,3], P2 %*% R[,3])
    ss = pmin(P1 %*% R[,4], P2 %*% R[,4])

    # prune invalid self joins (>1 bit per feature)
    I = matrix(1, nrow(P), 1);
    for( j in 1:ncol(foffb) ) {
      beg = foffb[1,j]+1;
      end = foffe[1,j];
      I = I & (rowSums(P[,beg:end]) <= 1);
    }
    P12 = P12[I,]
    P = P[I,]
    ss = as.matrix(ss[I])
    se = as.matrix(se[I])
    sm = as.matrix(sm[I])
    
    # prepare IDs for deduplication and pruning
    ID = matrix(0, nrow(P), 1);
    dom = foffe-foffb+1;
    for( j in 1:ncol(dom) ) {
      beg = foffb[1,j]+1;
      end = foffe[1,j];
      I = max.col(P[,beg:end],ties.method="last") * rowSums(P[,beg:end]);
      prod = 1;
      if(j<ncol(dom))
        prod = prod(dom[1,(j+1):ncol(dom)])
      ID = ID + I * prod;
    }

    # size pruning, with rowMin-rowMax transform 
    # to avoid densification (ignored zeros)
    map = table(ID, seq(1,nrow(P)))
    ex = matrix(1,nrow(map),1)
    ubSizes = 1/rowMaxs(map * (1/ex%*%t(ss)));
    ubSizes = as.matrix(replace(ubSizes, Inf, 0));
    fSizes = (ubSizes >= minSup)

    # error pruning
    ubError = 1/rowMaxs(map * (1/ex%*%t(se)));
    ubError = as.matrix(replace(ubError, Inf, 0));
    ubMError = 1/rowMaxs(map * (1/ex%*%t(sm)));
    ubMError = as.matrix(replace(ubMError, Inf, 0));
    ubScores = scoreUB(ubSizes, ubError, ubMError, eAvg, minSup, alpha, n2);
    TMP3 = analyzeTopK(TKC);
    minsc = TMP3[["minsc"]]
    fScores = (ubScores > minsc & ubScores > 0)

    # missing parents pruning
    numParents = rowSums((map %*% P12) != 0)
    fParents = (numParents == level);

    # apply all pruning
    map = map * ((fSizes & fScores & fParents) %*% matrix(1,1,ncol(map)));

    # deduplication of join outputs
    Dedup = as.matrix(map[rowMaxs(map)!=0,] != 0)
    P = (Dedup %*% P) != 0
  }
  return (P)
}

evalSlice = function(X, e, eAvg, tS, l, alpha) {
  I = (X %*% tS) == l;           # slice indicator
  ss = as.matrix(colSums(I));    # absolute slice size (nnz)
  se = as.matrix(t(t(e) %*% I)); # absolute slice error
  sm = as.matrix(colMaxs(I * e%*%matrix(1,1,ncol(I)))); # maximum tuple error in slice

  # score of relative error and relative size
  sc = score(ss, se, eAvg, alpha, nrow(X));
  R = as.matrix(cbind(sc, se, sm, ss));
  return (R)
}

decodeTopK = function(TK, foffb, foffe) {
  R = matrix(1, nrow(TK), ncol(foffb));
  if( nrow(TK) > 0 ) {
    for( j in 1:ncol(foffb) ) {
      beg = foffb[1,j]+1;
      end = foffe[1,j];
      I = rowSums(TK[,beg:end]) * max.col(TK[,beg:end],ties.method="last");
      R[, j] = I;
    }
  }
  return (R)
}

################################################################################
X = as.matrix(readMM(paste(args[1], "X.mtx", sep="")))
e = as.matrix(readMM(paste(args[1], "e.mtx", sep="")))
k = as.integer(args[2]);
tpEval = as.logical(args[3])

TMP = slicefinder(X=X, e=e, k=k, alpha=0.95, minSup=4, tpEval=tpEval, verbose=TRUE);
R = TMP[["TKC"]]

writeMM(as(R, "CsparseMatrix"), paste(args[4], "R", sep="")); 
