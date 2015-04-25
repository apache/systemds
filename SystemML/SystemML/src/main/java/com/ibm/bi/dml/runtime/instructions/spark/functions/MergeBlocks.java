package com.ibm.bi.dml.runtime.instructions.spark.functions;

import org.apache.spark.api.java.function.Function2;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.IJV;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.runtime.matrix.data.SparseRowsIterator;

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

		// ret = b1;
		// ret.merge(b2, false);

		if (b1.isInSparseFormat() && b2.isInSparseFormat()) {
			ret = mergeSparseBlocks(b1, b2);
		} else if (false == b1.isInSparseFormat()) {
			// b1 dense --> Merge b2 directly into a copy of b1, regardless
			// of whether it's dense or sparse
			ret = mergeIntoDenseBlock(b1, b2);
		} else {
			// b1 is not dense, b2 is dense --> Merge b1 into a copy of b2
			ret = mergeIntoDenseBlock(b2, b1);
		}

		// Sanity check
		if (ret.getNonZeros() != b1.getNonZeros() + b2.getNonZeros()) {
			throw new DMLRuntimeException("Number of non-zeros dont match: "
					+ ret.getNonZeros() + " " + b1.getNonZeros() + " "
					+ b2.getNonZeros());
		}

		return ret;
	}

	private MatrixBlock mergeSparseBlocks(MatrixBlock b1, MatrixBlock b2)
			throws DMLRuntimeException {

		// Validate inputs
		if (false == b1.isInSparseFormat()) {
			throw new DMLRuntimeException(
					"First block is not sparse in mergeSparseBlocks");
		}
		if (false == b2.isInSparseFormat()) {
			throw new DMLRuntimeException(
					"Second block is not sparse in mergeSparseBlocks");
		}

		if (b1.isEmpty()) {
			throw new DMLRuntimeException(
					"Empty block passed as first argument in mergeSparseBlocks");
		}
		if (b2.isEmpty()) {
			throw new DMLRuntimeException(
					"Empty block passed as second argument in mergeSparseBlocks");
		}

		// Construct merged output. Note shallow copy of rows.
		MatrixBlock ret = new MatrixBlock(b1.getNumRows(), b1.getNumColumns(),
				true);

		for (int r = 0; r < ret.getNumRows(); r++) {
			// Read directly from the internal representation
			SparseRow row1 = b1.getSparseRows()[r];

			SparseRow row2 = b2.getSparseRows()[r];

			if (null != row1 && null != row2) {
				// Both inputs have content for this row.
				SparseRow mergedRow = new SparseRow(row1);

				// TODO: Should we check for conflicting cells (O(nlogn)
				// overhead)?

				int[] indexes = row2.getIndexContainer();
				double[] values = row2.getValueContainer();

				for (int i = 0; i < indexes.length; i++) {
					mergedRow.append(indexes[i], values[i]);
				}

				mergedRow.sort();
				ret.appendRow(r, mergedRow);

				// throw new SystemMLException ("CONFLICTING_ROWS", r);
			} else if (null != row1) {
				// Input 1 has this row, input 2 does not
				ret.appendRow(r, row1);
			} else if (null != row2) {
				// Input 2 has this row, input 1 does not
				ret.appendRow(r, row2);
			} else {
				// Neither input has this row; do nothing
			}
		}

		return ret;
	}

	private MatrixBlock mergeIntoDenseBlock(MatrixBlock denseBlock,
			MatrixBlock otherBlock) throws DMLRuntimeException {
		if (denseBlock.isInSparseFormat()) {
			throw new DMLRuntimeException(
					"First block is not dens in mergeIntoDenseBlock");
		}

		// Start with the contents of the dense input
		MatrixBlock ret = new MatrixBlock(denseBlock.getNumRows(),
				denseBlock.getNumColumns(), false);
		ret.copy(denseBlock);

		// Add the contents of the other block.
		int numNonzerosAdded = 0;

		if (otherBlock.isInSparseFormat()) {
			// Other block is sparse, so we can directly access the nonzero
			// values.
			SparseRowsIterator itr = otherBlock.getSparseRowsIterator();
			while (itr.hasNext()) {
				IJV ijv = itr.next();

				// Sanity-check the previous value; the inputs to this
				// function shouldn't overlap
				double prevValue = ret.getValue(ijv.i, ijv.j);
				if (0.0D != prevValue) {
					throw new DMLRuntimeException(
							"NONZERO_VALUE_SHOULD_BE_ZERO");
					// throw new SystemMLException
					// ("NONZERO_VALUE_SHOULD_BE_ZERO", prevValue, ijv.i,
					// ijv.j, otherBlock, denseBlock); }
				}

				ret.setValue(ijv.i, ijv.j, ijv.v);
				numNonzerosAdded++;
			}
		} else {
			// Other block is dense; iterate over all values, adding
			// nonzeros.
			for (int r = 0; r < ret.getNumRows(); r++) {
				for (int c = 0; c < ret.getNumColumns(); c++) {
					double prevValue = ret.getValue(r, c);
					double otherValue = otherBlock.getValue(r, c);

					if (0.0D != otherValue) {
						if (0.0D != prevValue) {
							throw new DMLRuntimeException(
									"NONZERO_VALUE_SHOULD_BE_ZERO");
							// throw new SystemMLException
							// ("NONZERO_VALUE_SHOULD_BE_ZERO", prevValue,
							// ijv.i, ijv.j, otherBlock, denseBlock); }
						}

						// Use the "safe" accessor method, which also
						// updates sparsity information
						ret.setValue(r, c, otherValue);

						numNonzerosAdded++;
					}

				}
			}

		}

		// Sanity check
		if (numNonzerosAdded != otherBlock.getNonZeros()) {
			throw new DMLRuntimeException("Incorrect number of non-zeros");
			// throw new SystemMLException ("WRONG_NONZERO_COUNT",
			// numNonzerosAdded, otherBlock.getNonZeros (), otherBlock,
			// denseBlock); }
		}
		return ret;
	}

}