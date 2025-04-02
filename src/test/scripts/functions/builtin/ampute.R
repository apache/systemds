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
library("mice")

set.seed(args[3])
prop <- as.numeric(args[5])

col_names <- args[4] == "true"
X <- as.matrix(read.csv(args[1]), col_names=col_names)

# Create three patterns with different probabilities, each amputing a single variable:
numPatterns <- 3
freq <- rep(1 / numPatterns, numPatterns)
for (i in 1:numPatterns) {
  freq[i] <- i / 6
}
patterns <- matrix(1, nrow=numPatterns, ncol=ncol(X))
for (i in 1:numPatterns) {
  patterns[i, i] <- 0
}


res <- ampute(X, freq=freq, patterns=patterns, prop=prop)$amp


# Proportion of amputed rows:
amputed_rows <- apply(res, 1, function(row) any(is.na(row)))  # TRUE if row has missing values
proportion_amputed_rows <- mean(amputed_rows)
num_amputed_rows <- sum(amputed_rows)

# Pattern assigment proportions among amputed rows:
groupProps <- colSums(is.na(res)) / num_amputed_rows

# Print the result
cat("Proportion of total rows amputed (%): ", proportion_amputed_rows, "\n")
cat("Proportion of amputed rows by pattern (%): ", groupProps, "\n")

# Collect results, and output:
row_vector <- c(proportion_amputed_rows, groupProps)
# cat(row_vector)
writeMM(as(row_vector, "CsparseMatrix"), args[2])
