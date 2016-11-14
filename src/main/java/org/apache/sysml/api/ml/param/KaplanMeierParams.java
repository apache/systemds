package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;

public interface KaplanMeierParams extends HasAlpha, HasErrorType, HasCIType, HasTestType, HasTECol, HasGICol,
		HasSICol, HasTimestampIndex, HasEventIndex, HasGroupIndicesArray, HasGroupIndicesRangeStart,
		HasGroupIndicesRangeEnd, HasStratifyIndicesArray, HasStratifyIndicesRangeStart,
		HasStratifyIndicesRangeEnd, HasInputCol, HasOutputCol {

}
