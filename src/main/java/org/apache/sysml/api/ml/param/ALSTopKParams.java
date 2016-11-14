package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasFeaturesCol;
import org.apache.spark.ml.param.shared.HasMaxIter;
import org.apache.spark.ml.param.shared.HasThreshold;

public interface ALSTopKParams extends HasRank, HasReg, HasLambda, HasMaxIter, HasCheck, 
									   HasThreshold, HasK, HasFeaturesCol {

}
