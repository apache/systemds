
#install.packages("BiocManager")
#BiocManager::install("rhdf5")

args <- commandArgs(TRUE)

library("Matrix")
options(digits=22)

library("rhdf5")

Y = h5read(args[1],args[2],native = TRUE)
writeMM(as(Y, "CsparseMatrix"), paste(args[3], "Y", sep=""))
