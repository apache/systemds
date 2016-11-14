package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasMaxIter;
import org.apache.spark.ml.param.shared.HasRegParam;
import org.apache.spark.ml.param.shared.HasTol;

public interface LinearRegressionCGParams extends HasIntercept, HasRegParam, HasTol, HasMaxIter, 
												  HasDfam, HasVpow, HasLink, HasLpow, HasDisp {

}
