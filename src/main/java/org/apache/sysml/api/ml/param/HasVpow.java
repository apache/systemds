package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.DoubleParam;
import org.apache.spark.ml.param.Params;

public interface HasVpow extends Params {
	DoubleParam vpow();
	double getVpow();
}
