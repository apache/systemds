package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.List;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

public abstract class DataPartitioner {
	public abstract List<MatrixObject> doPartition(int k, MatrixObject mo);
}
