package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.Params;

public interface HasTECol extends Params {
	Param<String> teCol();
	String getTECol();
}
