package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;

public interface FeatureIndexerParams extends HasFeatureIndicesRangeStart, HasFeatureIndicesRangeEnd, 
//											  HasFeatureIndicesArray, HasInputCol, HasOutputCol {
											  HasInputCol, HasOutputCol {

}
