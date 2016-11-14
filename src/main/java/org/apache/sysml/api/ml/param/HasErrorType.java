package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.Params;

public interface HasErrorType extends Params {
	Param<String> errorType();
	String getErrorType();
}
