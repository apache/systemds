package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.LongParam;
import org.apache.spark.ml.param.Params;

public interface HasEventIndex extends Params {
	LongParam eventIndex();
	long getEventIndex();
}
