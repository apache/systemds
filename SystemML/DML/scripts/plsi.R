## data
V <- matrix( c(100, 10, 1,
               50 ,  5, 1,
               1  ,  5, 50,
               1  , 10, 100), c(4,3), byrow=T)


## set and save seed
set.seed(1)
save.seed <- .Random.seed

## print a separator line
hline <- function() {
    cat("--------------------------------------------------------------------------\n")
}

## determine a starting point
init <- function(nd, nw, nz) {
    D <- array(runif(nd*nz), c(nd,nz))
    D <- D %*% diag(1/colSums(D))
    Z <- array(runif(nz), nz)
    Z <- Z / sum(Z)
    W <- array(runif(nz*nw), c(nz,nw))
    W <- diag(1/rowSums(W)) %*% W
    list(D=D, Z=Z, W=W)
}

## Factorize using NMF and KL divergence: V = W*H + Eps
## Lee and Seung (1999) "Learning the parts of objects by non-negative matrix factorization"
## Requires t    H <- (diag(H$Z) %*% H$W) * Rwo scans per iteration. 
nmf.kl <- function(V, r=2, it=10, verbose=T, history=T, W=NULL, H=NULL) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    h <- list()
    
    ## start with random points & normalize (weird init to assure same starting point w/ all methods)
    if (is.null(W) | is.null(H)) {
        H <- init(nd, nw, nz)
        if (history) {
            h[[1]] <- list(it=0, D=H$D, Z=H$Z, W=H$W, R=R) 
        }
        W <- H$D
        H <- (diag(H$Z) %*% H$W) * R
    }

    ## solve NMF
    for (n in 1:it) {
        ## recompute W
        WH <- W %*% H
        Wp <- matrix(rep(0, nd*nz), c(nd,nz))
        for (d in 1:nd) {
            for (z in 1:nz) {
                Wp[d,z] <- W[d,z] * ( H[z,] %*% (V[d,]/WH[d,]) )
            }
        }
        Wp <- Wp %*% diag(1/colSums(Wp))
        W <- Wp

        ## recompute H
        WH <- W %*% H
        Hp <- matrix(rep(0, nz*nw), c(nz,nw))
        for (z in 1:nz) {
            for (w in 1:nw) {
                Hp[z,w] <- H[z,w] * ( W[,z] %*% (V[,w]/WH[,w]) )
            }
        }
        H <- Hp

        ## print status
        if (verbose) {
            hline()
            print(list(it=n, W=W, H=H))
        }

        # store current result
        if (history) {
            Z <- rowSums(H)/R
            h[[n+1]] <- list(it=n, D=W, Z=Z, W=diag(1/rowSums(H)) %*% H, R=R) 
        }
    }

    ## factor the result
    Z <- rowSums(H)/R
    H <- diag(1/rowSums(H)) %*% H

    ## return
    result <- list(it=n, D=W, Z=Z, W=H, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result <- c(result, list(history=h))
    }
    result
}

nmf.kl2 <- function(V, r=2, it=10, verbose=T, history=T) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    h <- list()
    
    ## start with random points & normalize (weird init to assure same starting point w/ all methods)
    H <- init(nd, nw, nz)
    if (history) {
        h[[1]] <- list(it=0, D=H$D, Z=H$Z, W=H$W, R=R) 
    }
    W <- H$D
    H <- (diag(H$Z) %*% H$W) * R

    ## solve NMF
    for (n in 1:it) {
        H <- diag(1/colSums(W)) %*% ( H * (t(W) %*% (V / (W %*% H)) ) )
        W <- ( W * ((V / (W %*% H)) %*% t(H) ) ) %*% diag(1/rowSums(H))

        ## print status
        if (verbose) {
            hline()
            print(list(it=n, W=W, H=H))
        }

        # store current result
        if (history) {
            Z <- rowSums(H)/R
            h[[n+1]] <- list(it=n, D=W, Z=Z, W=diag(1/rowSums(H)) %*% H, R=R) 
        }
    }

    ## factor the result
    Z <- rowSums(H)/R
    H <- diag(1/rowSums(H)) %*% H

    ## return
    result <- list(it=n, D=W, Z=Z, W=H, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result <- c(result, list(history=h))
    }
    result
}


## Factorize using NMF and KL divergence by using stochastic gradient descent
## Minimize objective function:
##                     f = sum_dw ( V_dw * (log(V_dw)-log(sum_z W_dz^2 * H_zw^2)) - V_dw + sum_z W_dz^2 * H_zw^2)
##              df/dW_dz = sum_dw 2*W_dz*H_zw^2 (-V_dw/(sum_z W_dz^2 * H_zw^2) + 1)
##              df/dH_zw = sum_dw 2*W_dz^2*H_zw (-V_dw/(sum_z W_dz^2 * H_zw^2) + 1)
nmf.kl.stochastic <- function(V, r=2, it=10, verbose=T, history=T, eps=0.1, delta=0.9) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    h <- list()
    
    ## start with random points & normalize (weird init to assure same starting point w/ all methods)
    H <- init(nd, nw, nz)
    if (history) {
        h[[1]] <- list(it=0, D=H$D, Z=H$Z, W=H$W, R=R) 
    }
    W <- sqrt(H$D)                          # solve for square roots of parameters
    H <- sqrt((diag(H$Z) %*% H$W))

    ## solve NMF
    for (n in 1:it) {
        ## perform a scan over the data
        for (d in 1:nd) {
            for (w in 1:nw) {
                s <- (W[d,]^2) %*% (H[,w]^2)
                dW.d <- 2*W[d,]*(H[,w]^2) * (-V[d,w]/R/s + 1)
                dH.w <- 2*(W[d,]^2)*H[,w] * (-V[d,w]/R/s + 1)
                W[d,] <- W[d,] - eps*(delta^(n-1))*dW.d
                H[,w] <- H[,w] - eps*(delta^(n-1))*dH.w
            }
        }
        
        ## renormalize
##        t <- sqrt(colSums(W^2))
##        W <- W %*% diag(1/t)
##        H <- diag(t) %*% H

        ## print status
        if (verbose) {
            hline()
            print(list(it=n, W=W^2, H=H^2))
        }

        # store current result
        if (history) {
            Z <- rowSums(H^2)
            h[[n+1]] <- list(it=n, D=W^2, Z=Z, W=diag(1/rowSums(H^2)) %*% (H^2), R=R) 
        }
    }

    ## factor the result
    Z <- rowSums(H^2)
    H <- diag(1/sqrt(rowSums(H^2))) %*% H

    ## return
    result <- list(it=n, D=W^2, Z=Z, W=H^2, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result <- c(result, list(history=h))
    }
    result
}


## Factorize using PLSA: V = D*Z*W + Eps, where
## - D: matrix containing entries of P(d|z)
## - Z: diagonal matrix that has P(z) on the diagonal
## - W: matrix containing entries of P(w|z)
## Hofmann (1999) "Probabilistic Latent Semantic Indexing"
## Requires two scans per iteration. 
plsa <- function(V, r=2, it=10, verbose=T, history=T) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    h <- list()

    ## create empty matrices
    Pd.z <- init(nd, nw, nz)
    Pz <- Pd.z$Z
    Pw.z <- Pd.z$W
    Pd.z <- t(Pd.z$D)
    if (history) {
        h[[1]] <- list(it=0, D=t(Pd.z), Z=Pz, W=Pw.z, R=R) 
    }
    Pz.dw <- array(rep(0, nd*nw*nz), c(nd,nw,nz))

    ## go
    for (n in 1:it) {
        ## E-step
        for (z in 1:nz) {
            Pz.dw[,,z] <- Pz[z] * Pd.z[z,] %*% t(Pw.z[z,])
        }

        ## normalize after E step
        for (d in 1:nd) {
            for (w in 1:nw) {
                Pz.dw[d,w,] <- Pz.dw[d,w,] / sum(Pz.dw[d,w,])
            }
        }

        ## M-step
        normalizer <- array(0, nz)
        for (w in 1:nw) {
            for (z in 1:nz) {
                Pw.z[z,w] <- V[,w] %*% Pz.dw[,w,z]
                normalizer[z] <- normalizer[z] + Pw.z[z,w]
            }
        }
        for (d in 1:nd) {
            for (z in 1:nz) {
                Pd.z[z,d] <- V[d,] %*% Pz.dw[d,,z]
            }
        }
        for (z in 1:nz) {
            Pz[z] <- 1/R * sum(V * Pz.dw[,,z])
            Pw.z[z,] <- Pw.z[z,] / normalizer[z]
            Pd.z[z,] <- Pd.z[z,] / normalizer[z]
        }

        ## print status
        if (verbose) {
            hline()
            print(list(it=n, Pz.dw=Pz.dw, Pd.z=Pd.z, Pw.z=Pw.z, Pz=Pz, R=R))
        }

        ## store current result
        if (history) {
            h[[n+1]] <- list(it=n, D=t(Pd.z), Z=Pz, W=Pw.z, R=R) 
        }
    }

    ## return
    result <- list(it=n, D=t(Pd.z), Z=Pz, W=Pw.z, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result <- c(result, list(history=h))
    }
    result
}

## Factorize using PLSA as above, but combine the E and the M step
## by inserting the E equations into the M equations.
## Requires one scan per iteration. 
plsa.optimized <- function(V, r=2, it=10, verbose=T, history=T) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    h <- list()

    ## create empty matrices
    Pd.z <- init(nd, nw, nz)
    Pz <- Pd.z$Z
    Pw.z <- Pd.z$W
    Pd.z <- t(Pd.z$D)
    if (history) {
        h[[1]] <- list(it=0, D=t(Pd.z), Z=Pz, W=Pw.z, R=R) 
    }

    ## go
    for (n in 1:it) {
        normalizer <- array(0, c(nd,nw)) ## sparse
        
        ## EM-step
        Nw.z <- array(0, c(nz,nw))
        Nd.z <- array(0, c(nz,nd))
        Nz <- array(0, nz)
        for (d in 1:nd) {
            for (w in 1:nw) {
                if (V[d,w] > 0) {
                    p <- array(0, nz)
                    for (z in 1:nz) {
                        p[z] <- Pz[z] * Pd.z[z,d] * Pw.z[z,w]
                    }
                    s <- sum(p)
                    p <- p/s
                    Nw.z[,w] <- Nw.z[,w] + V[d,w]*p
                    Nd.z[,d] <- Nd.z[,d] + V[d,w]*p
                    Nz <- Nz + V[d,w]*p
                }
            }
        }

        ## normalize
        for (z in 1:nz) {
            Nw.z[z,] <- Nw.z[z,] / Nz[z]
            Nd.z[z,] <- Nd.z[z,] / Nz[z]
            Nz[z] <- 1/R * Nz[z]
        }

        ## reassign
        Pd.z <- Nd.z
        Pw.z <- Nw.z
        Pz <- Nz

        ## print status
        if (verbose) {
            hline()
            print(list(it=n, Pd.z=Pd.z, Pw.z=Pw.z, Pz=Pz, R=R))
        }

        ## store current result
        if (history) {
            h[[n+1]] <- list(it=n, D=t(Pd.z), Z=Pz, W=Pw.z, R=R) 
        }
    }

    ## return
    result <- list(it=n, D=t(Pd.z), Z=Pz, W=Pw.z, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result <- c(result, list(history=h))
    }
    result
}

## Factorize using PLSA as in plsa.optimized, but modify the M step as follows:
## - update D
## - update W and Z using the new D (as opposed to the old D of the previous iteration)
## Requires two scans per iteration. Is identical to nmf.kl
plsa.nmf<- function(V, r=2, it=10, verbose=T, history=T) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    h <- list()

    ## create empty matrices
    Pd.z <- init(nd, nw, nz)
    Pz <- Pd.z$Z
    Pw.z <- Pd.z$W
    Pd.z <- t(Pd.z$D)
    if (history) {
        h[[1]] <- list(it=0, D=t(Pd.z), Z=Pz, W=Pw.z, R=R) 
    }

    ## go
    for (n in 1:it) {
        ## EM1-step (updates D)
        Nd.z <- array(0, c(nz,nd))
        Nz <- array(0, nz)
        for (d in 1:nd) {
            for (w in 1:nw) {
                if (V[d,w] > 0) {
                    p <- array(0, nz)
                    for (z in 1:nz) {
                        p[z] <- Pz[z] * Pd.z[z,d] * Pw.z[z,w]
                    }
                    s <- sum(p)
                    p <- p/s
                    Nd.z[,d] <- Nd.z[,d] + V[d,w]*p
                    Nz <- Nz + V[d,w]*p
                }
            }
        }
        for (z in 1:nz) {
            Nd.z[z,] <- Nd.z[z,] / Nz[z]
        }
        Pd.z <- Nd.z
        
        ## EM2-step (updates W, Z)
        Nw.z <- array(0, c(nz,nw))
        Nz <- array(0, nz)
        for (d in 1:nd) {
            for (w in 1:nw) {
                if (V[d,w] > 0) {
                    p <- array(0, nz)
                    for (z in 1:nz) {
                        p[z] <- Pz[z] * Pd.z[z,d] * Pw.z[z,w]
                    }
                    s <- sum(p)
                    p <- p/s
                    Nw.z[,w] <- Nw.z[,w] + V[d,w]*p
                    Nz <- Nz + V[d,w]*p
                }

            }
        }
        for (z in 1:nz) {
            Nw.z[z,] <- Nw.z[z,] / Nz[z]
            Nz[z] <- 1/R * Nz[z]
        }
        Pw.z <- Nw.z
        Pz <- Nz

        ## print status
        if (verbose) {
            hline()
            print(list(it=n, Pd.z=Pd.z, Pw.z=Pw.z, Pz=Pz, R=R))
        }

        ## store current result
        if (history) {
            h[[n+1]] <- list(it=n, D=t(Pd.z), Z=Pz, W=Pw.z, R=R) 
        }
    }

    ## return
    result <- list(it=n, D=t(Pd.z), Z=Pz, W=Pw.z, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result <- c(result, list(history=h))
    }
    result
}

kl <- function(P, Q) {
    sum(P*(log(P)-log(Q)) - P + Q)
}
yy

plot.kl <- function(..., type="b", ylim=NULL) {
    data <- list(...)
    n <- length(data)
    R <- sum(V)

    y <- list()
    min <- NA
    max <- NA
    it <- NA
    for (i in 1:n) {
        y[[i]] <- sapply(data[[i]]$history, function(x) kl(V/R, x$D%*%diag(x$Z)%*%x$W))
        m <- min(y[[i]])
        if (is.na(min) | m<min) {
            min <- m
        }
        m <- max(y[[i]])
        if (is.na(max) | m>max) {
            max <- m
        }
        m <- length(y[[i]])
        if (is.na(it) | m>it) {
            it <- m
        }
    }

    if (is.null(ylim)) ylim <- c(min, max)
    plot(NA, type="n", xlim=c(1,it), ylim=ylim, log="y")
    for (i in 1:n) {
        lines(1:length(y[[i]]), y[[i]], type=type, pch=i)
    }
    legend("topright", legend=names(data), pch=1:n)

    invisible(y)
}





f <- function(x, V, r, verbose=T, history=T) {
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)
    W <- matrix(x[1:(nd*nz)], c(nd, nz))
    H <- matrix(x[(nd*nz+1):(nd*nz+nz*nw)], c(nz, nw))

    if (history) {
        ## renormalize
        t <- sqrt(colSums(W^2))
        W <- W %*% diag(1/t)
        H <- diag(t) %*% H

        ## print status
        if (verbose) {
            hline()
            print(list(it=length(nmf.optim.h), W=W^2, H=H^2))
        }

        # store current result
        if (history) {
            Z <- colSums(W^2) * rowSums(H^2)
            nmf.optim.h[[length(nmf.optim.h)+1]] <<- list(it=length(nmf.optim.h), D=W^2 %*% diag(1/colSums(W^2)), Z=Z, W=diag(1/rowSums(H^2)) %*% (H^2), R=R)
        }
    }

    kl(V/R, W^2 %*% H^2)
}

df <- function(x, V, r, ...) {
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)
    W <- matrix(x[1:(nd*nz)], c(nd, nz))
    H <- matrix(x[(nd*nz+1):(nd*nz+nz*nw)], c(nz, nw))

    dW <- matrix(rep(0, nd*nz), c(nd, nz))
    dH <- matrix(rep(0, nz*nw), c(nz, nw))
    for (d in 1:nd) {
        for (w in 1:nw) {
            s <- (W[d,]^2) %*% (H[,w]^2)
            dW[d,] <- dW[d,] + 2*W[d,]*(H[,w]^2) * (-V[d,w]/R/s + 1)
            dH[,w] <- dH[,w] + 2*(W[d,]^2)*H[,w] * (-V[d,w]/R/s + 1)
        }
    }

    c(as.array(dW), as.array(dH))
}

nmf.optim.h <<- c()
nmf.optim <- function(V, r=2, method="L-BFGS-B", verbose=T, history=T) {
    ## initialize
    nd <- nrow(V)
    nw <- ncol(V)
    nz <- r
    R <- sum(V)    
    .Random.seed <<- save.seed
    nmf.optim.h <<- list()
    
    ## start with random points & normalize (weird init to assure same starting point w/ all methods)
    H <- init(nd, nw, nz)
    if (history) {
        nmf.optim.h[[1]] <<- list(it=0, D=H$D, Z=H$Z, W=H$W, R=R) 
    }
    W <- sqrt(H$D)                          # solve for square roots of parameters
    H <- sqrt((diag(H$Z) %*% H$W))

    x <- optim(c(as.array(W), as.array(H)), f, gr=df, V, 2, verbose, history,  method=method)

    W <- matrix(x$par[1:(nd*nz)], c(nd, nz))
    H <- matrix(x$par[(nd*nz+1):(nd*nz+nz*nw)], c(nz, nw))

    ## factor the result
    Z <- colSums(W^2) * rowSums(H^2)
    W <- W %*% diag(1/sqrt(colSums(W^2)))
    H <- diag(1/sqrt(rowSums(H^2))) %*% H

    ## return
    result <- list(it=50, D=W^2, Z=Z, W=H^2, R=R)
    if (verbose) {
        hline()
        print(result)
    }
    if (history) {
        result$history <- nmf.optim.h
    }
    result
}


## compare all methods (picked the best case for stocahstic gradient descent)
V.nmf.kl <- nmf.kl(V, it=80)

V.nmf.kl2 <- nmf.kl2(V, it=80)

x <- plot.kl(NMF=V.nmf.kl, NMF2=V.nmf.kl2)

V.nmf.kl.stochastic <- nmf.kl.stochastic(V, eps=0.4, delta=0.99, it=80)

V.plsa <- plsa(V, it=80)

V.plsa.optimized <- plsa.optimized(V, it=80)

V.plsa.nmf <- plsa.nmf(V, it=80)

V.nmf.lbfgs <- nmf.optim(V)

V.nmf.bfgs <- nmf.optim(V, method="BFGS")

V.nmf.cg <- nmf.optim(V, method="CG")

x <- plot.kl(NMF=V.nmf.kl, "NMF (stochastic)" = V.nmf.kl.stochastic, PLSA=V.plsa, "PLSA (optimized)" = V.plsa.optimized, "PLSA (NMF-like)" = V.plsa.nmf, "BFGS" = V.nmf.bfgs, "L-BFGS" = V.nmf.lbfgs, "Conjugate Gradient" = V.nmf.cg); x

## try various parameters for gradient descent
l <- list()
for (eps in c(0.1, 0.2, 0.4, 0.5)) {
    for (delta in c(0.9, 0.95, 0.99)) {
        name <- paste(eps, delta)
        l[[name]] <- nmf.kl.stochastic(V, eps=eps, delta=delta, it=50, verbose=F)
    }
}

l[["type"]]="b"
do.call("plot.kl", l)
