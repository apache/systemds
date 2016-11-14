package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.Params;

public interface HasFeatureIndicesRangeEnd extends Params {
	IntParam featureIndicesRangeEnd();
	int getFeatureIndicesRangeEnd();
}
