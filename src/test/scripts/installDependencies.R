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

print("Starting install RScripts")

args <- commandArgs(TRUE)

options(repos=structure(c(CRAN="http://cran.r-project.org")))

custom_install <- function(pkg) {
    if(!is.element(pkg, installed.packages()[,1])) {
		# Installing to temp folder, if you want to permenently install change lib path
		if (length(args)==0) {
 			install.packages(pkg);
		} else if (length(args) == 1){
			install.packages(pkg, lib= args[1]);
		}
	}
} 

list_user_pkgs <- function() {
	print("List of user installed packages:")

	ip <- as.data.frame(installed.packages()[,c(1,3:4)])
	rownames(ip) <- NULL
	ip <- ip[is.na(ip$Priority),1:2,drop=FALSE]
	print(ip, row.names=FALSE)
}

custom_install("Matrix");
custom_install("psych");
custom_install("moments");
custom_install("boot");
custom_install("matrixStats");
custom_install("outliers");
custom_install("caret");
custom_install("sigmoid");
custom_install("DescTools");
custom_install("mice");
custom_install("mclust");
custom_install("dbscan");
custom_install("imputeTS");
custom_install("FNN");
custom_install("class");
custom_install("unbalanced");
custom_install("naivebayes");
custom_install("BiocManager");
custom_install("mltools");
custom_install("einsum");
BiocManager::install("rhdf5");

print("Installation Done")


# supply any two parameters to list all user installed packages
# e.g. "sudo Rscript installDependencies.R a b"
if (length(args) == 2) {
	list_user_pkgs()
}
