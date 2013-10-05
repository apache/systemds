#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

args <- commandArgs(TRUE)
options(digits=22)

library("Matrix")
library("moments")

A1 <- readMM(paste(args[1], "D.mtx", sep=""))
K1 <- readMM(paste(args[1], "K.mtx", sep=""))
A <- as.matrix(A1);
K <- as.matrix(K1);
maxC = args[2];  


# number of features/attributes
n = ncol(A);

# number of data records
m = nrow(A);
m

# number of statistics
numBaseStats = 17; # (14 scale stats, 3 categorical stats)

max_kind = max(K);
  
# matrices to store computed statistics
baseStats = array(0,dim=c(numBaseStats,n)); 

if (maxC > 0) {
  countsArray = array(0,dim=c(maxC,n)); 
}
  
for(i in 1:n) {

	# project out the i^th column
	F = as.matrix(A[,i]);

	kind = K[1,i];

	if ( kind == 1 ) {
		print("scale");
		# compute SCALE statistics on the projected column
		minimum = min(F);
		maximum = max(F);
		rng = maximum - minimum;

		mu = mean(F);
		m2 = moment(F, order=2, central=TRUE);
		m3 = moment(F, order=3, central=TRUE);
		m4 = moment(F, order=4, central=TRUE);

		var = m/(m-1.0)*m2;
    
		std_dev = sqrt(var);
		se = std_dev/sqrt(m);
		cv = std_dev/mu;

		g1 = m3/(std_dev^3);
		g2 = m4/(std_dev^4) - 3;
		#se_g1=sqrt( 6*m*(m-1.0) / ((m-2.0)*(m+1.0)*(m+3.0)) ); 
		se_g1=sqrt( (6/(m-2.0)) * (m/(m+1.0)) * ((m-1.0)/(m+3.0)) ); 

		#se_g2= sqrt( (4*(m^2-1)*se_g1^2)/((m+5.0)*(m-3.0)) );  
		se_g2=sqrt( (4/(m+5.0)) * ((m^2-1)/(m-3.0)) * se_g1^2 ); 

		md = quantile(F, 0.5, type = 1);
		iqm = mean( subset(F, F>quantile(F,1/4,type = 1) & F<=quantile(F,3/4,type = 1) ) )
    
		# place the computed statistics in output matrices
		baseStats[1,i] = minimum;
		baseStats[2,i] = maximum;
		baseStats[3,i] = rng;

		baseStats[4,i] = mu;
		baseStats[5,i] = var;
		baseStats[6,i] = std_dev;
		baseStats[7,i] = se;
		baseStats[8,i] = cv;

		baseStats[9,i] = g1;
		baseStats[10,i] = g2;
		baseStats[11,i] = se_g1;
		baseStats[12,i] = se_g2;

		baseStats[13,i] = md;
		baseStats[14,i] = iqm;
	}
	else {
		if (kind == 2 | kind == 3) {
			print("categorical");
			
			# check if the categorical column has valid values
			minF = min(F);
			if (minF <=0) {
				print("ERROR: Categorical attributes can only take values starting from 1.");
			}
			else {
				# compute CATEGORICAL statistics on the projected column
				cat_counts = table(F);  # counts for each category
				num_cat = nrow(cat_counts); # number of categories

        mx = max(t(as.vector(cat_counts)))
        mode = which(cat_counts == mx)    
        
      	numModes = length(cat_counts[ cat_counts==mx ]);

				# place the computed statistics in output matrices
				baseStats[15,i] = num_cat;
				baseStats[16,i] = mode;
				baseStats[17,i] = numModes;

        if (max_kind > 1) {
				  countsArray[1:length(cat_counts),i] = cat_counts;
				}
			}
		}
	}
}

writeMM(as(baseStats, "CsparseMatrix"), paste(args[3], "base.stats", sep=""));
if (max_kind > 1) {
  writeMM(as(countsArray, "CsparseMatrix"), paste(args[3], "categorical.counts", sep=""));
}

