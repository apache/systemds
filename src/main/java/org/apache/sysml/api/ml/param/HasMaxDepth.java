package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.IntParam;
import org.apache.spark.ml.param.Params;

public interface HasMaxDepth extends Params {
	IntParam maxDepth();
	int getMaxDepth();
}
