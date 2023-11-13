/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysds.runtime.compress.colgroup;

import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IdentityDictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public abstract class ADictBasedColGroup extends AColGroupCompressed implements IContainADictionary {
	private static final long serialVersionUID = -3737025296618703668L;
	/** Distinct value tuples associated with individual bitmaps. */
	protected final IDictionary _dict;

	/**
	 * A Abstract class for column groups that contain IDictionary for values.
	 * 
	 * @param colIndices The Column indexes
	 * @param dict       The dictionary to contain the distinct tuples
	 */
	protected ADictBasedColGroup(IColIndex colIndices, IDictionary dict) {
		super(colIndices);
		_dict = dict;
		if(dict == null)
			throw new NullPointerException("null dict is invalid");

	}

	public IDictionary getDictionary() {
		return _dict;
	}

	@Override
	public final void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		if(_dict instanceof IdentityDictionary) {

			final MatrixBlockDictionary md = ((IdentityDictionary) _dict).getMBDict();
			final MatrixBlock mb = md.getMatrixBlock();
			// The dictionary is never empty.
			if(mb.isInSparseFormat())
				decompressToDenseBlockSparseDictionary(db, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, mb.getDenseBlockValues());
		}
		else if(_dict instanceof MatrixBlockDictionary) {
			final MatrixBlockDictionary md = (MatrixBlockDictionary) _dict;
			final MatrixBlock mb = md.getMatrixBlock();
			// The dictionary is never empty.
			if(mb.isInSparseFormat())
				decompressToDenseBlockSparseDictionary(db, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, mb.getDenseBlockValues());
		}
		else
			decompressToDenseBlockDenseDictionary(db, rl, ru, offR, offC, _dict.getValues());
	}

	@Override
	public final void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
		if(_dict instanceof IdentityDictionary) {

			final MatrixBlockDictionary md = ((IdentityDictionary) _dict).getMBDict();
			final MatrixBlock mb = md.getMatrixBlock();
			// The dictionary is never empty.
			if(mb.isInSparseFormat())
				decompressToSparseBlockSparseDictionary(sb, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToSparseBlockDenseDictionary(sb, rl, ru, offR, offC, mb.getDenseBlockValues());
		}
		else if(_dict instanceof MatrixBlockDictionary) {
			final MatrixBlockDictionary md = (MatrixBlockDictionary) _dict;
			final MatrixBlock mb = md.getMatrixBlock();
			// The dictionary is never empty.
			if(mb.isInSparseFormat())
				decompressToSparseBlockSparseDictionary(sb, rl, ru, offR, offC, mb.getSparseBlock());
			else
				decompressToSparseBlockDenseDictionary(sb, rl, ru, offR, offC, mb.getDenseBlockValues());
		}
		else
			decompressToSparseBlockDenseDictionary(sb, rl, ru, offR, offC, _dict.getValues());
	}

	/**
	 * Decompress to DenseBlock using a sparse dictionary to lookup into.
	 * 
	 * @param db   The dense db block to decompress into
	 * @param rl   The row to start decompression from
	 * @param ru   The row to end decompression at
	 * @param offR The row offset to insert into
	 * @param offC The column offset to insert into
	 * @param sb   The sparse dictionary block to take value tuples from
	 */
	protected abstract void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb);

	/**
	 * Decompress to DenseBlock using a dense dictionary to lookup into.
	 * 
	 * @param db     The dense db block to decompress into
	 * @param rl     The row to start decompression from
	 * @param ru     The row to end decompression at
	 * @param offR   The row offset to insert into
	 * @param offC   The column offset to insert into
	 * @param values The dense dictionary values, linearized row major.
	 */
	protected abstract void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values);

	/**
	 * Decompress to SparseBlock using a sparse dictionary to lookup into.
	 * 
	 * @param ret  The dense ret block to decompress into
	 * @param rl   The row to start decompression from
	 * @param ru   The row to end decompression at
	 * @param offR The row offset to insert into
	 * @param offC The column offset to insert into
	 * @param sb   The sparse dictionary block to take value tuples from
	 */
	protected abstract void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb);

	/**
	 * Decompress to SparseBlock using a dense dictionary to lookup into.
	 * 
	 * @param ret    The dense ret block to decompress into
	 * @param rl     The row to start decompression from
	 * @param ru     The row to end decompression at
	 * @param offR   The row offset to insert into
	 * @param offC   The column offset to insert into
	 * @param values The dense dictionary values, linearized row major.
	 */
	protected abstract void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values);

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		_dict.write(out);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += _dict.getExactSizeOnDisk();
		return ret;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += _dict.getInMemorySize();
		size += 8; // dict reference
		return size;
	}

	@Override
	public final AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols) {
		if(right.isEmpty())
			return null;

		final int nCol = right.getNumColumns();
		// make sure allCols is allocated
		allCols = allCols == null ? ColIndexFactory.create(nCol) : allCols;

		final IColIndex agCols = (right.isInSparseFormat()) ? // find Cols
			rightMMGetColsSparse(right.getSparseBlock(), nCol, allCols) : // sparse
			rightMMGetColsDense(right.getDenseBlockValues(), nCol, allCols, right.getNonZeros()); // dense

		if(agCols == null)
			return null;

		final int nVals = getNumValues();
		final IDictionary preAgg = (right.isInSparseFormat()) ? // Chose Sparse or Dense
			rightMMPreAggSparse(nVals, right.getSparseBlock(), agCols, 0, nCol) : // sparse
			_dict.preaggValuesFromDense(nVals, _colIndexes, agCols, right.getDenseBlockValues(), nCol); // dense
		return allocateRightMultiplication(right, agCols, preAgg);
	}

	protected abstract AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes,
		IDictionary preAgg);

	/**
	 * Find the minimum number of columns that are effected by the right multiplication
	 * 
	 * @param b       The dense values in the right matrix
	 * @param nCols   The max number of columns in the right matrix
	 * @param allCols The all columns int list
	 * @param nnz     The number of non zero values in b
	 * @return a list of the column indexes effected in the output column group
	 */
	protected IColIndex rightMMGetColsDense(double[] b, final int nCols, IColIndex allCols, long nnz) {
		if(nCols > 200 || nnz > (b.length * 0.7)) // just return the int array
			return allCols;
		else { // try to do the best we can
			Set<Integer> aggregateColumnsSet = new HashSet<>();

			for(int k = 0; k < _colIndexes.size(); k++) {
				int rowIdxOffset = _colIndexes.get(k) * nCols;
				for(int h = 0; h < nCols; h++)
					if(b[rowIdxOffset + h] != 0.0) {
						aggregateColumnsSet.add(h);
						continue;
					}

			}
			if(aggregateColumnsSet.size() == nCols)
				return allCols;
			if(aggregateColumnsSet.size() == 0)
				return null;

			int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
			Arrays.sort(aggregateColumns);
			return ColIndexFactory.create(aggregateColumns);
		}
	}

	/**
	 * Find the minimum number of columns that are effected by the right multiplication
	 * 
	 * @param b       The sparse matrix on the right
	 * @param retCols The number of columns contained in the sparse matrix.
	 * @return a list of the column indexes effected in the output column group
	 */
	protected IColIndex rightMMGetColsSparse(SparseBlock b, int retCols, IColIndex allCols) {
		Set<Integer> aggregateColumnsSet = new HashSet<>();

		for(int h = 0; h < _colIndexes.size(); h++) {
			int colIdx = _colIndexes.get(h);
			if(!b.isEmpty(colIdx)) {
				int[] sIndexes = b.indexes(colIdx);
				for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++)
					aggregateColumnsSet.add(sIndexes[i]);
			}
			if(aggregateColumnsSet.size() == retCols)
				return allCols;
		}
		if(aggregateColumnsSet.size() == 0)
			return null;

		int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
		Arrays.sort(aggregateColumns);
		return ColIndexFactory.create(aggregateColumns);
	}

	private IDictionary rightMMPreAggSparse(int numVals, SparseBlock b, IColIndex aggregateColumns, int cl, int cu) {
		final double[] ret = new double[numVals * aggregateColumns.size()];
		for(int h = 0; h < _colIndexes.size(); h++) {
			final int colIdx = _colIndexes.get(h);
			if(b.isEmpty(colIdx))
				continue;

			final double[] sValues = b.values(colIdx);
			final int[] sIndexes = b.indexes(colIdx);
			int retIdx = 0;
			for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++) {
				while(aggregateColumns.get(retIdx) < sIndexes[i])
					retIdx++;
				// It is known in this case that the sIndex always correspond to the aggregateColumns.
				// if(sIndexes[i] == aggregateColumns[retIdx])
				for(int j = 0, offOrg = h;
					j < numVals * aggregateColumns.size();
					j += aggregateColumns.size(), offOrg += _colIndexes.size()) {
					ret[j + retIdx] += _dict.getValue(offOrg) * sValues[i];
				}
			}

		}
		return Dictionary.create(ret);
	}

	@Override
	public final AColGroup copyAndSet(IColIndex colIndexes) {
		return copyAndSet(colIndexes, _dict);
	}

	/**
	 * This method copies the column group and sets the dictionary to another dictionary. The method shallow copies
	 * underlying data structures.
	 * 
	 * NOTE: Be very carefull with this since it invalidate the contracts maintained via standard compression.
	 * 
	 * @param newDictionary A new dictionary to point to.
	 * @return A new shallow copy of the column group
	 */
	public final AColGroup copyAndSet(IDictionary newDictionary) {
		return copyAndSet(_colIndexes, newDictionary);
	}

	protected abstract AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary);

}
