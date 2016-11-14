package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasTol;

public interface CoxRegressionParams extends HasAlpha, HasTol, HasMaxInnerIter, HasMaxOuterIter, HasTECol, HasFCol,
		HasRCol, HasFeatureIndicesRangeStart, HasFeatureIndicesRangeEnd, HasFeatureIndicesArray,
		HasTimestampIndex, HasEventIndex {

}
