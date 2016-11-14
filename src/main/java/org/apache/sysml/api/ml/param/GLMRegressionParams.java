package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasRegParam;
import org.apache.spark.ml.param.shared.HasTol;

public interface GLMRegressionParams extends HasIntercept, HasRegParam, HasTol, HasYNeg, HasMaxInnerIter, HasMaxOuterIter,
											 HasDfam, HasVpow, HasLink, HasLpow, HasDisp {

}
