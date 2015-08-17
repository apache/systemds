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

library("Matrix")

factorial <- function(arr, pos){
	if(pos == 1){
     arr[1, pos] = 1
	} else {
		arr = factorial(arr, pos-1)
		arr[1, pos] = pos * arr[1, pos-1]
	}

  return(arr);	
}

n = as.integer(args[1])
arr = matrix(0, 1, n)
arr = factorial(arr, n)

R = matrix(0, 1, n);
for(i in 1:n)
{
   R[1,i] = arr[1, i];
}

writeMM(as(R, "CsparseMatrix"), paste(args[2], "R", sep="")); 


