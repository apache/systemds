package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasThreshold;
import org.apache.spark.ml.param.shared.HasTol;

public interface StepGLMRegressionParams extends HasIntercept, HasTol, HasYNeg, HasThreshold, HasMaxInnerIter, 
												 HasMaxOuterIter, HasDfam, HasVpow, HasLink, HasLpow, HasDisp {

}
