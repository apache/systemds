args <- commandArgs(TRUE)

library("Matrix")
library("imputeTS")

input_matrix = as.matrix(readMM(paste(args[1], "A.mtx", sep="")))
input_matrix[input_matrix==0] = NA

bins_in = as.numeric(args[2])

Out = statsNA(input_matrix, bins = bins_in, print_only = FALSE)

Out1=Out["length_series"][[1]]
Out2=Out["number_NAs"][[1]]
Out3=Out["number_na_gaps"][[1]]
Out4=Out["average_size_na_gaps"][[1]]
Out5=as.numeric(sub("%","",Out["percentage_NAs"][[1]],fixed=TRUE))/100
Out6=Out["longest_na_gap"][[1]]
Out7=Out["most_frequent_na_gap"][[1]]
Out8=Out["most_weighty_na_gap"][[1]]

write(Out1, paste(args[3], "Out1", sep=""))
write(Out2, paste(args[3], "Out2", sep=""))
write(Out3, paste(args[3], "Out3", sep=""))
write(Out4, paste(args[3], "Out4", sep=""))
write(Out5, paste(args[3], "Out5", sep=""))
write(Out6, paste(args[3], "Out6", sep=""))
write(Out7, paste(args[3], "Out7", sep=""))
write(Out8, paste(args[3], "Out8", sep=""))