custom_install <- function(pkg) {
    if(!is.element(pkg, installed.packages()[,1])) {
 		install.packages(pkg, repos="http://cran.stat.ucla.edu/");
	}
} 

custom_install("Matrix");
custom_install("plotrix");
custom_install("psych");
custom_install("moments");
custom_install("batch");
custom_install("matrixStats");