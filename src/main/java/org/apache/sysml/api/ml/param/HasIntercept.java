package org.apache.sysml.api.ml.param;

import org.apache.spark.ml.param.IntParam;

public interface HasIntercept {
	IntParam intercept();
	int getIntercept();
}
