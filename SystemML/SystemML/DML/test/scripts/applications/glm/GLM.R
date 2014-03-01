#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2014
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.applications.GLMTest.java
# Intended to solve GLM Regression using R, in order to compare against the DML implementation
# INPUT 1: Matrix X [rows, columns]
# INPUT 2: Matrix y [rows, 1]
# INPUT 3-6: Distribution family and link, see below:
# ---------------------------------------------
#   Dst Var Lnk Lnk   Distribution       Cano-
#   typ pow typ pow   Family.link        nical?
# ---------------------------------------------
#    1  0.0  1 -1.0   Gaussian.inverse
#    1  0.0  1  0.0   Gaussian.log
#    1  0.0  1  1.0   Gaussian.id         Yes
#    1  1.0  1  0.0   Poisson.log         Yes
#    1  1.0  1  0.5   Poisson.sqrt
#    1  1.0  1  1.0   Poisson.id
#    1  2.0  1 -1.0   Gamma.inverse       Yes
#    1  2.0  1  0.0   Gamma.log
#    1  2.0  1  1.0   Gamma.id
#    1  3.0  1 -2.0   InvGaussian.1/mu^2  Yes
#    1  3.0  1 -1.0   InvGaussian.inverse
#    1  3.0  1  0.0   InvGaussian.log
#    1  3.0  1  1.0   InvGaussian.id
#    1   *   1   *    AnyVariance.AnyLink
# ---------------------------------------------
#    2 -1.0  *   *    Binomial {-1, 1}
#    2  0.0  *   *    Binomial { 0, 1}
#    2  1.0  *   *    Binomial two-column
#    2   *   1  0.0   Binomial.log
#    2   *   2   *    Binomial.logit      Yes
#    2   *   3   *    Binomial.probit
#    2   *   4   *    Binomial.cloglog
#    2   *   5   *    Binomial.cauchit
# ---------------------------------------------
# INPUT 3: (int) Distribution type
# INPUT 4: (double) For Power families: Variance power of the mean
# INPUT 5: (int) Link function type
# INPUT 6: (double) Link as power of the mean
# INPUT 7: (double) tolerance (epsilon)
# INPUT 8: the regression coefficients output file
# OUTPUT : Matrix beta [columns, 1]
#
# Assume that $GLMR_HOME is set to the home of the R script
# Assume input and output directories are $GLMR_HOME/in/ and $GLMR_HOME/expected/
# Rscript $GLMR_HOME/GLM.R $GLMR_HOME/in/X.mtx $GLMR_HOME/in/y.mtx 2 0.0 2 0.0 0.00000001 $GLMR_HOME/expected/w.mtx

args <- commandArgs (TRUE);

library ("Matrix");
# library ("batch");

options (warn = -1);

X_here <- readMM (args[1]);  # (paste (args[1], "X.mtx", sep=""));
y_here <- readMM (args[2]);  # (paste (args[1], "y.mtx", sep=""));

num_records  <- nrow (X_here);
num_features <- ncol (X_here);
dist_type  <- as.integer (args[3]);
dist_param <- as.numeric (args[4]);
link_type  <- as.integer (args[5]);
link_power <- as.numeric (args[6]);
eps_n <- as.numeric (args[7]);

f_ly <- gaussian ();
var_power <- dist_param;

if (dist_type == 1 & var_power == 0.0 & link_type == 1 & link_power ==  1.0) { f_ly <- gaussian (link = "identity");         } else
if (dist_type == 1 & var_power == 0.0 & link_type == 1 & link_power == -1.0) { f_ly <- gaussian (link = "inverse");          } else
if (dist_type == 1 & var_power == 0.0 & link_type == 1 & link_power ==  0.0) { f_ly <- gaussian (link = "log");              } else
if (dist_type == 1 & var_power == 1.0 & link_type == 1 & link_power ==  1.0) { f_ly <-  poisson (link = "identity");         } else
if (dist_type == 1 & var_power == 1.0 & link_type == 1 & link_power ==  0.0) { f_ly <-  poisson (link = "log");              } else
if (dist_type == 1 & var_power == 1.0 & link_type == 1 & link_power ==  0.5) { f_ly <-  poisson (link = "sqrt");             } else
if (dist_type == 1 & var_power == 2.0 & link_type == 1 & link_power ==  1.0) { f_ly <-    Gamma (link = "identity");         } else
if (dist_type == 1 & var_power == 2.0 & link_type == 1 & link_power == -1.0) { f_ly <-    Gamma (link = "inverse");          } else
if (dist_type == 1 & var_power == 2.0 & link_type == 1 & link_power ==  0.0) { f_ly <-    Gamma (link = "log");              } else
if (dist_type == 1 & var_power == 3.0 & link_type == 1 & link_power ==  1.0) { f_ly <- inverse.gaussian (link = "identity"); } else
if (dist_type == 1 & var_power == 3.0 & link_type == 1 & link_power == -1.0) { f_ly <- inverse.gaussian (link = "inverse");  } else
if (dist_type == 1 & var_power == 3.0 & link_type == 1 & link_power ==  0.0) { f_ly <- inverse.gaussian (link = "log");      } else
if (dist_type == 1 & var_power == 3.0 & link_type == 1 & link_power == -2.0) { f_ly <- inverse.gaussian (link = "1/mu^2");   } else
if (dist_type == 2                    & link_type == 1 & link_power ==  0.0) { f_ly <- binomial (link = "log");              } else
if (dist_type == 2                    & link_type == 2                     ) { f_ly <- binomial (link = "logit");            } else
if (dist_type == 2                    & link_type == 3                     ) { f_ly <- binomial (link = "probit");           } else
if (dist_type == 2                    & link_type == 4                     ) { f_ly <- binomial (link = "cloglog");          } else
if (dist_type == 2                    & link_type == 5                     ) { f_ly <- binomial (link = "cauchit");          }

# quasi(link = "identity", variance = "constant")
# quasibinomial(link = "logit")
# quasipoisson(link = "log")

if (dist_type == 2 & dist_param != 1.0) {
    y_here <- (y_here - dist_param) / (1.0 - dist_param);
}

# epsilon 	tolerance: the iterations converge when |dev - devold|/(|dev| + 0.1) < epsilon.
# maxit 	integer giving the maximal number of IWLS iterations.
# trace 	logical indicating if output should be produced for each iteration.
#
c_rol <- glm.control (epsilon = eps_n, maxit = 100, trace = FALSE);

X_matrix = as.matrix (X_here);
y_matrix = as.matrix (y_here);

glmOut <- glm (y_matrix ~ X_matrix - 1, family = f_ly, control = c_rol);
betas <- coef (glmOut);
print (c("Deviance", glmOut$deviance));
writeMM (as (betas, "CsparseMatrix"), args[8], format = "text");

