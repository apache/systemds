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

import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Special sideways compressed column group not supposed to be used outside of the compressed transform encode.
 */
public class ColGroupUncompressedArray extends AColGroup {
	private static final long serialVersionUID = -825423333043292199L;

	public final Array<?> array;
	public final int id; // columnID

	public ColGroupUncompressedArray(Array<?> data, int id, IColIndex colIndexes) {
		super(colIndexes);
		this.array = data;
		this.id = id;
	}

	@Override
	public int getNumValues() {
		return array.size();
	}

	@Override
	public long estimateInMemorySize() {
		// not accurate estimate, but guaranteed larger.
		return MatrixBlock.estimateSizeInMemory(array.size(), 1, array.size()) + 80;
	}

	@Override
	public String toString() {
		return "UncompressedArrayGroup: " + id + " " + _colIndexes;
	}

	@Override
	public AColGroup copyAndSet(IColIndex colIndexes) {
		return new ColGroupUncompressedArray(array, id, colIndexes);	
	}

	@Override
	public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {
		throw new UnsupportedOperationException("Unimplemented method 'decompressToDenseBlockTransposed'");
	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {
		throw new UnsupportedOperationException("Unimplemented method 'decompressToSparseBlockTransposed'");
	}

	@Override
	public double getIdx(int r, int colIdx) {
		throw new UnsupportedOperationException("Unimplemented method 'getIdx'");
	}

	@Override
	public CompressionType getCompType() {
		throw new UnsupportedOperationException("Unimplemented method 'getCompType'");
	}

	@Override
	protected ColGroupType getColGroupType() {
		throw new UnsupportedOperationException("Unimplemented method 'getColGroupType'");
	}

	@Override
	public void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC) {
		throw new UnsupportedOperationException("Unimplemented method 'decompressToDenseBlock'");
	}

	@Override
	public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
		throw new UnsupportedOperationException("Unimplemented method 'decompressToSparseBlock'");
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
		throw new UnsupportedOperationException("Unimplemented method 'rightMultByMatrix'");
	}

	@Override
	public void tsmm(MatrixBlock ret, int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'tsmm'");
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		throw new UnsupportedOperationException("Unimplemented method 'leftMultByMatrixNoPreAgg'");
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'leftMultByAColGroup'");
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new UnsupportedOperationException("Unimplemented method 'tsmmAColGroup'");
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		throw new UnsupportedOperationException("Unimplemented method 'scalarOperation'");
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new UnsupportedOperationException("Unimplemented method 'binaryRowOpLeft'");
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new UnsupportedOperationException("Unimplemented method 'binaryRowOpRight'");
	}

	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int nRows, int rl, int ru) {
		throw new UnsupportedOperationException("Unimplemented method 'unaryAggregateOperations'");
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		throw new UnsupportedOperationException("Unimplemented method 'sliceSingleColumn'");
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		throw new UnsupportedOperationException("Unimplemented method 'sliceMultiColumns'");
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		throw new UnsupportedOperationException("Unimplemented method 'sliceRows'");
	}

	@Override
	public double getMin() {
		throw new UnsupportedOperationException("Unimplemented method 'getMin'");
	}

	@Override
	public double getMax() {
		throw new UnsupportedOperationException("Unimplemented method 'getMax'");
	}

	@Override
	public double getSum(int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'getSum'");
	}

	@Override
	public boolean containsValue(double pattern) {
		throw new UnsupportedOperationException("Unimplemented method 'containsValue'");
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'getNumberNonZeros'");
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		throw new UnsupportedOperationException("Unimplemented method 'replace'");
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'computeColSums'");
	}

	@Override
	public CmCovObject centralMoment(CMOperator op, int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'centralMoment'");
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'rexpandCols'");
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		throw new UnsupportedOperationException("Unimplemented method 'getCost'");
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		throw new UnsupportedOperationException("Unimplemented method 'unaryOperation'");
	}

	@Override
	public boolean isEmpty() {
		throw new UnsupportedOperationException("Unimplemented method 'isEmpty'");
	}

	@Override
	public AColGroup append(AColGroup g) {
		throw new UnsupportedOperationException("Unimplemented method 'append'");
	}

	@Override
	protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
		throw new UnsupportedOperationException("Unimplemented method 'appendNInternal'");
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		throw new UnsupportedOperationException("Unimplemented method 'getCompressionScheme'");
	}

	@Override
	public AColGroup recompress() {
		throw new UnsupportedOperationException("Unimplemented method 'recompress'");
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		throw new UnsupportedOperationException("Unimplemented method 'getCompressionInfo'");
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		throw new UnsupportedOperationException("Unimplemented method 'fixColIndexes'");
	}

	@Override
	public AColGroup reduceCols() {
		throw new UnsupportedOperationException("Unimplemented method 'reduceCols'");
	}

	@Override
	public double getSparsity() {
		throw new UnsupportedOperationException("Unimplemented method 'getSparsity'");
	}

	@Override
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new UnsupportedOperationException("Unimplemented method 'sparseSelection'");
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new UnsupportedOperationException("Unimplemented method 'denseSelection'");
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		throw new UnsupportedOperationException("Unimplemented method 'splitReshape'");
	}

}
