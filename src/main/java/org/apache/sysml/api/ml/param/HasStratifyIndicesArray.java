package org.apache.sysml.api.ml.param;

import java.util.List;

//import org.apache.spark.ml.param.IntArrayParam;
import org.apache.spark.ml.param.Params;

public interface HasStratifyIndicesArray extends Params {
//	IntArrayParam stratifyIndices();
	List<Integer> getStratifyIndices();
}
