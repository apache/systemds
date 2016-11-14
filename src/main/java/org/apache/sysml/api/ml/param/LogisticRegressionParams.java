package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasRegParam;
import org.apache.spark.ml.param.shared.HasTol;

public interface LogisticRegressionParams extends HasIntercept, HasRegParam, HasTol, HasMaxOuterIter, 
												  HasMaxInnerIter, HasDfam, HasVpow, HasLink, HasLpow, 
												  HasDisp {
	
}
