package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.Params;

public interface HasTestType extends Params {
	Param<String> testType();
	String getTestType();
}
