package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.Params;

public interface HasSICol extends Params {
	Param<String> siCol();
	String getSICol();
}