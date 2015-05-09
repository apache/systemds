package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;

public class MergeBlocks implements Function2<MatrixBlock, MatrixBlock, MatrixBlock> {

	private static final long serialVersionUID = -8881019027250258850L;

	@Override
	public MatrixBlock call(MatrixBlock b1, MatrixBlock b2) throws Exception {
		MatrixBlock ret = null;
		if (b1.getNumRows() != b2.getNumRows()
				|| b1.getNumColumns() != b2.getNumColumns()) {
			throw new DMLRuntimeException("Mismatched block sizes: "
					+ b1.getNumRows() + " " + b1.getNumColumns() + " "
					+ b2.getNumRows() + " " + b2.getNumColumns());
		}

		boolean isB1Empty = b1.isEmpty();
		boolean isB2Empty = b2.isEmpty();

		if (isB1Empty && !isB2Empty) {
			return b2; // b2.clone();
		} else if (!isB1Empty && isB2Empty) {
			return b1;
		} else if (isB1Empty && isB2Empty) {
			return b1;
		}

		// TODO: Is it ok to merge into sparse format if  b1 is not dense, b2 is dense ?
		ret = new MatrixBlock(b1);
		ret.merge(b2, false);

		// Sanity check
		if (ret.getNonZeros() != b1.getNonZeros() + b2.getNonZeros()) {
			throw new DMLRuntimeException("Number of non-zeros dont match: "
					+ ret.getNonZeros() + " " + b1.getNonZeros() + " "
					+ b2.getNonZeros());
		}

		return ret;
	}

}