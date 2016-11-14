package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.BooleanParam;
import org.apache.spark.ml.param.Params;

public interface HasCheck extends Params {
	BooleanParam check();
	boolean getCheck();
}
