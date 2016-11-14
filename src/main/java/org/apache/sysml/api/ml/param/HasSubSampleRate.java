package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.Params;

public interface HasSubSampleRate extends Params {
	DoubleParam subSampleRate();
	double getSubSampleRate();
}
