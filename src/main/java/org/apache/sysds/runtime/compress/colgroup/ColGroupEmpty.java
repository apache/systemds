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

import java.io.DataInput;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IIterate;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetEmpty;
import org.apache.sysds.runtime.compress.colgroup.scheme.EmptyScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.estim.encoding.EncodingFactory;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CM_COV_Object;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

public class ColGroupEmpty extends AColGroupCompressed
	implements IContainADictionary, IContainDefaultTuple, AOffsetsGroup, IMapToDataGroup {
	private static final long serialVersionUID = -2307677253622099958L;

	/**
	 * Constructs an Constant Colum Group, that contains only one tuple, with the given value.
	 * 
	 * @param colIndices The Colum indexes for the column group.
	 */
	public ColGroupEmpty(IColIndex colIndices) {
		super(colIndices);
	}

	public static ColGroupEmpty create(int nCol) {
		return new ColGroupEmpty(ColIndexFactory.create(nCol));
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.EMPTY;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.EMPTY;
	}

	@Override
	public void decompressToDenseBlock(DenseBlock target, int rl, int ru, int offR, int offC) {
		// do nothing.
	}

	@Override
	public void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC) {
		// do nothing.
	}

	@Override
	public void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru) {
		// do nothing.
	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut) {
		// do nothing.
	}

	@Override
	public double getIdx(int r, int colIdx) {
		return 0;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double v = op.executeScalar(0);
		if(v == 0)
			return this;
		double[] retV = new double[_colIndexes.size()];
		Arrays.fill(retV, v);
		return ColGroupConst.create(_colIndexes, Dictionary.create(retV));
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double v = op.fn.execute(0);
		if(v == 0)
			return this;
		double[] retV = new double[_colIndexes.size()];
		Arrays.fill(retV, v);
		return ColGroupConst.create(_colIndexes, Dictionary.create(retV));
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe)
			return this;
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_colIndexes.size()];
		final int lenV = _colIndexes.size();
		for(int i = 0; i < lenV; i++)
			retVals[i] = fn.execute(v[_colIndexes.get(i)], 0);
		return ColGroupConst.create(_colIndexes, Dictionary.create(retVals));
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		if(isRowSafe)
			return this;
		final ValueFunction fn = op.fn;
		final double[] retVals = new double[_colIndexes.size()];
		final int lenV = _colIndexes.size();
		for(int i = 0; i < lenV; i++)
			retVals[i] = fn.execute(0, v[_colIndexes.get(i)]);
		return ColGroupConst.create(_colIndexes, Dictionary.create(retVals));
	}

	@Override
	public int getNumValues() {
		return 0;
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock c, int nRows) {
		// do nothing, but should never be called
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		// do nothing, but should never be called
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		// do nothing, but should never be called
	}

	@Override
	public boolean containsValue(double pattern) {
		return pattern == 0;
	}

	@Override
	public long getNumberNonZeros(int nRows) {
		return 0;
	}

	@Override
	protected AColGroup sliceSingleColumn(int idx) {
		return new ColGroupEmpty(ColIndexFactory.create(1));
	}

	@Override
	protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
		return new ColGroupEmpty(outputCols);
	}

	@Override
	public AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k) {
		return null;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		if(pattern == 0)
			return ColGroupConst.create(_colIndexes, replace);
		else
			return new ColGroupEmpty(_colIndexes);
	}

	@Override
	public final double getMin() {
		return 0;
	}

	@Override
	public final double getMax() {
		return 0;
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		return builtin.execute(c, 0);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		IIterate it = _colIndexes.iterator();
		while(it.hasNext()) {
			final int colId = it.next();
			c[colId] = builtin.execute(c[colId], 0);
		}
	}

	@Override
	protected void computeSum(double[] c, int nRows) {
		// do nothing
	}

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		// do nothing
	}

	@Override
	public void computeColSums(double[] c, int nRows) {
		// do nothing
	}

	@Override
	protected void computeSumSq(double[] c, int nRows) {
		// do nothing
	}

	@Override
	protected void computeColSumsSq(double[] c, int nRows) {
		// do nothing
	}

	@Override
	protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		for(int r = rl; r < ru; r++)
			c[r] = builtin.execute(c[r], 0);
	}

	@Override
	protected void tsmm(double[] result, int numColumns, int nRows) {
		// do nothing
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		c[0] = 0;
	}

	@Override
	protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		for(int i = 0; i < c.length; i++)
			c[i] = 0;
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		for(int i = 0; i < c.length; i++)
			c[i] = 0;
	}

	@Override
	protected double[] preAggSumRows() {
		return null;
	}

	@Override
	protected double[] preAggSumSqRows() {
		return null;
	}

	@Override
	protected double[] preAggProductRows() {
		return null;
	}

	@Override
	protected double[] preAggBuiltinRows(Builtin builtin) {
		return null;
	}

	@Override
	public CM_COV_Object centralMoment(CMOperator op, int nRows) {
		CM_COV_Object ret = new CM_COV_Object();
		op.fn.execute(ret, 0.0, nRows);
		return ret;
	}

	@Override
	public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
		if(!ignore)
			throw new DMLRuntimeException(
				"Invalid input to rexpand since it contains zero use ignore flag to encode anyway");
		else
			return create(max);
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nCols = getNumCols();
		return e.getCost(nRows, 1, nCols, 1, 0.00001);
	}

	@Override
	public boolean isEmpty() {
		return true;
	}

	public static ColGroupEmpty read(DataInput in) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		return new ColGroupEmpty(cols);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		return null;
	}

	@Override
	public AColGroup copyAndSet(IColIndex colIndexes) {
		return new ColGroupEmpty(colIndexes);
	}

	@Override
	public AColGroup append(AColGroup g) {
		if(g instanceof ColGroupEmpty && g._colIndexes.size() == _colIndexes.size())
			return this;
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		for(int i = 0; i < g.length; i++) {
			final AColGroup gs = g[i];
			if(!_colIndexes.equals(gs._colIndexes))
				throw new DMLCompressionException("Invalid columns not matching " + gs._colIndexes + " " + _colIndexes);
			if(gs instanceof ColGroupEmpty)
				continue;
			else
				return gs.appendNInternal(g, blen, rlen);
		}
		return this;
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return EmptyScheme.create(this);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		EstimationFactors ef = new EstimationFactors(getNumValues(), 1, 0, 0.0);
		return new CompressedSizeInfoColGroup(_colIndexes, ef, estimateInMemorySize(), CompressionType.EMPTY,
			getEncoding());
	}

	@Override
	public IEncode getEncoding() {
		return EncodingFactory.create(this);
	}

	@Override
	public ADictionary getDictionary() {
		return null;
	}

	@Override
	public double[] getDefaultTuple() {
		return new double[getNumCols()];
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		return that instanceof ColGroupEmpty || that instanceof ColGroupConst;
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		return new ColGroupEmpty(newColIndex);
	}

	@Override
	public AOffset getOffsets() {
		return new OffsetEmpty();
	}

	@Override
	public AMapToData getMapToData() {
		return MapToFactory.create(0, 0);
	}

	@Override
	public AColGroup reduceCols() {
		return null;
	}

	@Override
	public double getSparsity() {
		return 0.0;
	}
	
	@Override
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, AColGroup right) {

		if(!(right instanceof ColGroupEmpty))
			throw new NotImplementedException("Combine on Empty column only allowing empty column groups");

		IColIndex combIndex = _colIndexes.combine(right.getColIndices().shift(nCol));

		return new ColGroupEmpty(combIndex);

	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		final int s = _colIndexes.size();
		final int[] newColumns = new int[s * multiplier];
		for(int i = 0; i < multiplier; i++)
			for(int j = 0; j < s; j++)
				newColumns[i * s + j] = _colIndexes.get(j) + nColOrg * i;

		return new AColGroup[]{new ColGroupEmpty(ColIndexFactory.create(newColumns))};
	}

	@Override
	public AColGroup combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		for(AColGroup g : right) {
			if(!(g instanceof ColGroupEmpty))
				throw new NotImplementedException("Combine on Empty column only allowing empty column groups");
		}

		IColIndex combinedIndex = _colIndexes;
		int i = 0;
		for(AColGroup g : right) {
			i += 1;
			combinedIndex = combinedIndex.combine(g.getColIndices().shift(nCol * i));
		}

		return new ColGroupEmpty(combinedIndex);
	}
}
