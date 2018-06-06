package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.ArrayList;
import java.util.List;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;

/**
 * Disjoint_Contiguous data partitioner:
 *
 * for each worker, use a right indexing
 * operation X[beg:end,] to obtain contiguous,
 * non-overlapping partitions of rows.
 */
public class DataPartitionerDC extends DataPartitioner {
	@Override
	public List<MatrixObject> doPartition(int k, MatrixObject mo) {
		List<MatrixObject> list = new ArrayList<>();
		long stepSize = (long) Math.ceil(mo.getNumRows() / k);
		long begin = 1;
		while (begin < mo.getNumRows()) {
			long end = Math.min(begin + stepSize, mo.getNumRows());
			MatrixObject pmo = ParamservUtils.sliceMatrix(mo, begin, end);
			list.add(pmo);
			begin = end + 1;
		}
		return list;
	}
}
