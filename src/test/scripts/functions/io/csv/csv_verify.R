#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#-------------------------------------------------------------



args <- commandArgs(TRUE)
options(digits=22)

library(Matrix);

# R interprets "0" as a categorical ("factor") value, so we need to read the
# file in as strings and convert everything to numeric explicitly.
A = read.csv(args[1], header=FALSE, stringsAsFactors=FALSE);
A = sapply(A, as.numeric);
x = sum(A);
write(x, args[2]);

