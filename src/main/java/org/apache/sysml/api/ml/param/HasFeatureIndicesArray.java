package org.apache.sysml.api.ml.param;

import java.util.List;

//import org.apache.spark.ml.param.IntArrayParam;
import org.apache.spark.ml.param.Params;

public interface HasFeatureIndicesArray extends Params {
//	IntArrayParam featureIndices();
	List<Integer> getFeatureIndices();
}
