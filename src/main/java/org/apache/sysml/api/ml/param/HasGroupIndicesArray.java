package org.apache.sysml.api.ml.param;

import java.util.List;

//import org.apache.spark.ml.param.IntArrayParam;
import org.apache.spark.ml.param.Params;

public interface HasGroupIndicesArray extends Params {
//	IntArrayParam groupIndices();
	List<Integer> getGroupIndices();
}
