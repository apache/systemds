#-------------------------------------------------------------
# IBM Confidential
# OCO Source Materials
# (C) Copyright IBM Corp. 2010, 2013
# The source code for this program is not published or
# otherwise divested of its trade secrets, irrespective of
# what has been deposited with the U.S. Copyright Office.
#-------------------------------------------------------------

# JUnit test class: dml.test.integration.applications.GLMReg.java
# Intended to solve GLM Regression using R, in order to compare against the DML implementation
# INPUT 1: Matrix X [rows, columns]
# INPUT 2: Matrix y [rows, 1]
# INPUT 3: distribution_family: integer 00 ... 99, see the table below:
# -------------------------------------------------------------------------
#          FAMILY:   0x        1x        2x        3x        4x        9x
#   LINK:          Gaussian  Binomial  Poisson   Gamma  inv_Gaussian  Other
#   x0 = identity    00*       --        20        30        40        --
#   x1 = inverse     01        --        --        31*       41        --
#   x2 = log         02        12        22*       32        42        --
#   x3 = sqrt        --        --        23        --        --        --
#   x4 = 1/mu^2      --        --        --        --        44*       --
#   x5 = logit       --        15*       --        --        --        --
#   x6 = probit      --        16        --        --        --        --
#   x7 = cauchit     --        17        --        --        --        --
#   x8 = cloglog     --        18        --        --        --        --
#   x9 = other      (09)      (19)      (29)      (39)      (49)      (99)
#   (Here * denotes the natural link, () means unfinished implementation)
#   (99 = Use variance-power and link-power constants specified elsewhere)
# -------------------------------------------------------------------------
# INPUT 4: tolerance (epsilon)
# INPUT 5: the regression coefficients output file
# OUTPUT : Matrix beta [columns, 1]
#
# Assume that $GLMR_HOME is set to the home of the R script
# Assume input and output directories are $GLMR_HOME/in/ and $GLMR_HOME/expected/
# Assume distribution_family = 10, epsilon = 0.00000001
# Rscript $GLMR_HOME/GLMReg.R $GLMR_HOME/in/X.mtx $GLMR_HOME/in/y.mtx 10      0.00000001 $GLMR_HOME/expected/w.mtx
#                             args[1]             args[2]             args[3] args[4]    args[5]

args <- commandArgs (TRUE);

library ("Matrix");
# library ("batch");

options (warn = -1);

X_here <- readMM (args[1]);  # (paste (args[1], "X.mtx", sep=""));
y_here <- readMM (args[2]);  # (paste (args[1], "y.mtx", sep=""));

num_records <- nrow (X_here);
num_features <- ncol (X_here);
distribution_family <- as.integer (args[3]);
eps_n <- as.numeric (args[4]);

f_ly <- gaussian ();

if (distribution_family == 00) { f_ly <- gaussian (link = "identity");         } else
if (distribution_family == 01) { f_ly <- gaussian (link = "inverse");          } else
if (distribution_family == 02) { f_ly <- gaussian (link = "log");              } else
if (distribution_family == 12) { f_ly <- binomial (link = "log");              } else
if (distribution_family == 15) { f_ly <- binomial (link = "logit");            } else
if (distribution_family == 16) { f_ly <- binomial (link = "probit");           } else
if (distribution_family == 17) { f_ly <- binomial (link = "cauchit");          } else
if (distribution_family == 18) { f_ly <- binomial (link = "cloglog");          } else
if (distribution_family == 20) { f_ly <-  poisson (link = "identity");         } else
if (distribution_family == 22) { f_ly <-  poisson (link = "log");              } else
if (distribution_family == 23) { f_ly <-  poisson (link = "sqrt");             } else
if (distribution_family == 30) { f_ly <-    Gamma (link = "identity");         } else
if (distribution_family == 31) { f_ly <-    Gamma (link = "inverse");          } else
if (distribution_family == 32) { f_ly <-    Gamma (link = "log");              } else
if (distribution_family == 40) { f_ly <- inverse.gaussian (link = "identity"); } else
if (distribution_family == 41) { f_ly <- inverse.gaussian (link = "inverse");  } else
if (distribution_family == 42) { f_ly <- inverse.gaussian (link = "log");      } else
if (distribution_family == 44) { f_ly <- inverse.gaussian (link = "1/mu^2");   }

# quasi(link = "identity", variance = "constant")
# quasibinomial(link = "logit")
# quasipoisson(link = "log")

if (10 <= distribution_family && distribution_family <= 19) {
    y_here <- (y_here + 1) / 2;
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

writeMM (as (betas, "CsparseMatrix"), args[5], format = "text");

