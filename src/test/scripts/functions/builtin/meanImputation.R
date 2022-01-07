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
library("DescTools")


Salaries <- read.csv(args[1], header=TRUE, na.strings = "19")

mode = Mode(Salaries$yrs.since.phd, na.rm = TRUE)

Salaries$yrs.since.phd[is.na(Salaries$yrs.since.phd)]<-mode

t = Salaries$yrs.service
t[is.na(t)]<-0
mean = mean(t)

Salaries$yrs.service[is.na(Salaries$yrs.service)]<-mean
output = cbind(Salaries$yrs.since.phd, Salaries$yrs.service)

Mode <- function(x, na.rm = FALSE) {
  if(na.rm){
    x = x[!is.na(x)]
  }

  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

writeMM(as(output, "CsparseMatrix"), paste(args[2], "B", sep=""));
