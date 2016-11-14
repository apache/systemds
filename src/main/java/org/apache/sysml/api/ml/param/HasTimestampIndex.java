package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.LongParam;
import org.apache.spark.ml.param.Params;

public interface HasTimestampIndex extends Params {
	LongParam timestampIndex();
	long getTimestampIndex();
}
