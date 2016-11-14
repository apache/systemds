package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;

public interface TimestampEventSelectorParams extends HasTimestampIndex, HasEventIndex, HasInputCol, HasOutputCol {

}
