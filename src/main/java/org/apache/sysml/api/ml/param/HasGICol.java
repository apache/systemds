package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.Params;

public interface HasGICol extends Params {
	Param<String> giCol();
	String getGICol();
}