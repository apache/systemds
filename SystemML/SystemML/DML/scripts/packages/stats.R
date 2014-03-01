

# ---------------------------------------------------------------
# R functions for Descriptive Statistics
# TODO: must add functions for univariate stats
# ---------------------------------------------------------------

# Categorical-Categorical
bivar_cc = function(DS, i, j) {
  options(digits=10);
  F = table(DS[,i], DS[,j]);
  cst = chisq.test(F);

  # get the chi-squared coefficient from the list
  chi_squared = as.numeric(cst[1]);
  pValue = as.numeric(cst[3]);

  q = min(dim(F));
  W = sum(F);
  cramers_v = sqrt(chi_squared/(W*(q-1)));

  cat("pValue = ", pValue, "\n");
  cat("cramersV = ", cramers_v, "\n");
}

#-------------------------------------------------

# Scale-Categorical
bivar_sc = function(DS, i, j) {
    
    Yv = DS[,i];
    Av = DS[,j];

    W = nrow(DS);
    my = mean(Yv); #sum(Yv)/W;
    varY = var(Yv);

    CFreqs = as.matrix(table(Av)); 
    CMeans = as.matrix(aggregate(Yv, by=list(Av), "mean")$x);
    CVars = as.matrix(aggregate(Yv, by=list(Av), "var")$x);

    # number of categories
    R = nrow(CFreqs);

    Eta = sqrt(1 - ( sum((CFreqs-1)*CVars) / ((W-1)*varY) ));

    anova_num = sum( (CFreqs*(CMeans-my)^2) )/(R-1);
    anova_den = sum( (CFreqs-1)*CVars )/(W-R);
    AnovaF = anova_num/anova_den;

    cat("Eta = ", Eta, "\n");
    cat("AnovaF = ", AnovaF, "\n");
}

#-------------------------------------------------

#Scale-Scale
bivar_ss = function(DS, i, j) {
    # cor.test returns a list containing t-statistic, df, p-value, and R
    cort = cor.test(DS[,i], DS[,j]);
    R = as.numeric(cort[4]);

    cat("Pearson R = ", R, "\n");
}

#-------------------------------------------------

