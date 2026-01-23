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

package org.apache.sysds.test.component.compress.colgroup;

import static org.junit.Assert.fail;

import java.util.concurrent.ExecutorService;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.DMLScriptException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.AColGroupCompressed;
import org.apache.sysds.runtime.compress.colgroup.ADictBasedColGroup;
import org.apache.sysds.runtime.compress.colgroup.APreAgg;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupRLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCSingleZeros;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCZeros;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.lib.CLALibLeftMultBy;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class ColGroupNegativeTests {

	@Test(expected = DMLCompressionException.class)
	public void testFailingGroupCompressionSingleThread() {
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		ColGroupFactory.compressColGroups(new FakeMatrixBlock(), new FakeCompressedSizeInfo(), cs, 1);
	}

	@Test(expected = DMLCompressionException.class)
	public void testFailingGroupCompressionParallel() {
		CompressionSettings cs = new CompressionSettingsBuilder().create();
		ColGroupFactory.compressColGroups(new FakeMatrixBlock(), new FakeCompressedSizeInfo(), cs, 10);
	}

	@Test(expected = NullPointerException.class)
	public void testFailingDictColGroup() {
		new FakeDictBasedColGroup();
	}

	@Test(expected = DMLScriptException.class)
	public void preAggInvalidValueFunction() {
		AColGroupCompressed g = (AColGroupCompressed) ColGroupConst.create(new double[] {1, 2, 3, 4, 5});
		g.preAggRows(new FakeValueFunction());
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidIndexingSum() {
		invalidIndexing(new AggregateOperator(1, Plus.getPlusFnObject()));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidIndexingSumSq() {
		invalidIndexing(new AggregateOperator(1, KahanPlusSq.getKahanPlusSqFnObject()));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidIndexingProd() {
		invalidIndexing(new AggregateOperator(1, Multiply.getMultiplyFnObject()));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidIndexingMax() {
		invalidIndexing(new AggregateOperator(1, Builtin.getBuiltinFnObject(BuiltinCode.MAX)));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidIndexingCumMax() {
		invalidIndexing(new AggregateOperator(1, Builtin.getBuiltinFnObject(BuiltinCode.CUMMAX)));
	}

	public void invalidIndexing(AggregateOperator ag) {
		AColGroupCompressed g = (AColGroupCompressed) ColGroupConst.create(new double[] {1, 2, 3, 4, 5});
		g.unaryAggregateOperations(new AggregateUnaryOperator(ag, new FakeIndexing()), new double[5], 10, 0, 10);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidPreAggregateClass() {
		try {

			MatrixBlock mb = TestUtils.ceil(TestUtils.generateTestMatrixBlock(1, 10, 1, 3, 0.3, 134));
			APreAgg a = (APreAgg) ColGroupTest.getColGroup(mb, CompressionType.DDC, 10);
			a.preAggregateThatIndexStructure(new FakeAPreAgg());
		}
		catch(DMLRuntimeException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to throw exception");
		}
	}

	@Test(expected = Exception.class)
	public void invalidEmptyRowSum() {
		CLALibLeftMultBy.rowSum(new MatrixBlock(10, 10, true), 0, 10, 0, 10);
	}

	private class FakeIndexing extends IndexFunction {
		private static final long serialVersionUID = -4099420257856761251L;

		protected FakeIndexing() {
		}

		@Override
		public void execute(MatrixIndexes in, MatrixIndexes out) {
		}

		@Override
		public void execute(CellIndex in, CellIndex out) {
		}

		@Override
		public boolean computeDimension(int row, int col, CellIndex retDim) {
			return false;
		}

		@Override
		public boolean computeDimension(DataCharacteristics in, DataCharacteristics out) {
			return false;
		}
	}

	private class FakeMatrixBlock extends MatrixBlock {
		protected FakeMatrixBlock() {
			super(1, 1, 10.0);
		}
	}

	private class FakeCompressedSizeInfo extends CompressedSizeInfo {
		protected FakeCompressedSizeInfo() {
			super((CompressedSizeInfoColGroup) null);
		}
	}

	private class FakeValueFunction extends ValueFunction {
		private static final long serialVersionUID = -585186573175954738L;

		private FakeValueFunction() {

		}
	}

	private class FakeAPreAgg extends APreAgg {
		private static final long serialVersionUID = 8759470530917794282L;

		private FakeAPreAgg() {
			super(ColIndexFactory.create(1), Dictionary.createNoCheck(new double[13]), null);
		}

		@Override
		public void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu) {

		}

		@Override
		public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {

		}

		@Override
		protected void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {

		}

		@Override
		protected void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {

		}

		@Override
		protected void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {

		}

		@Override
		public boolean sameIndexStructure(AColGroupCompressed that) {
			return false;
		}

		@Override
		protected int numRowsToMultiply() {
			return 0;
		}

		@Override
		public int[] getCounts(int[] out) {
			return null;
		}

		@Override
		protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
			SparseBlock sb) {

		}

		@Override
		protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
			double[] values) {

		}

		@Override
		protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
			SparseBlock sb) {

		}

		@Override
		protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
			double[] values) {

		}

		@Override
		protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {

		}

		@Override
		protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {

		}

		@Override
		protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {

		}

		@Override
		public double getIdx(int r, int colIdx) {
			return 0;
		}

		@Override
		public CompressionType getCompType() {
			return null;
		}

		@Override
		protected ColGroupType getColGroupType() {
			return null;
		}

		@Override
		public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {

		}

		@Override
		public AColGroup scalarOperation(ScalarOperator op) {
			return null;
		}

		@Override
		public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
			return null;
		}

		@Override
		public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
			return null;
		}

		@Override
		public double getCost(ComputationCostEstimator e, int nRows) {
			return 0;
		}

		@Override
		public AColGroup unaryOperation(UnaryOperator op) {
			return null;
		}

		@Override
		protected double computeMxx(double c, Builtin builtin) {
			// TODO Auto-generated method stub
			return 0;
		}

		@Override
		protected void computeColMxx(double[] c, Builtin builtin) {
			// TODO Auto-generated method stub

		}

		@Override
		public boolean containsValue(double pattern) {
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
			// TODO Auto-generated method stub

		}

		@Override
		public AColGroup sliceRows(int rl, int ru) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AColGroup append(AColGroup g) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public ICLAScheme getCompressionScheme() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AColGroup recompress() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'recompress'");
		}

		@Override
		public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'getCompressionInfo'");
		}

		@Override
		protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'fixColIndexes'");
		}

		@Override
		protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock dict) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToDenseBlockTransposedSparseDictionary'");
		}

		@Override
		protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToDenseBlockTransposedDenseDictionary'");
		}

		@Override
		protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock dict, int nColOut) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToSparseBlockTransposedSparseDictionary'");
		}

		@Override
		protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToSparseBlockTransposedDenseDictionary'");
		}

		@Override
		public void leftMMIdentityPreAggregateDense(MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl, int cu) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'leftMMIdentityPreAggregateDense'");
		}

		@Override
		public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'sparseSelection'");
		}

		@Override
		protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'denseSelection'");
		}
		
		@Override 
		public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'splitReshape'");
		}

		@Override
		protected boolean allowShallowIdentityRightMult() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'allowShallowIdentityRightMult'");
		}

		@Override
		public AColGroup[] splitReshapePushDown(int multiplier, int nRow, int nColOrg, ExecutorService pool)
			throws Exception {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'splitReshapePushDown'");
		}
	}

	private class FakeDictBasedColGroup extends ADictBasedColGroup {
		private static final long serialVersionUID = 7578204757649117273L;

		private FakeDictBasedColGroup() {
			super(null, null);
		}

		@Override
		protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
			SparseBlock sb) {
		}

		@Override
		protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
			double[] values) {
		}

		@Override
		protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
			SparseBlock sb) {
		}

		@Override
		protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
			double[] values) {
		}

		@Override
		protected double computeMxx(double c, Builtin builtin) {
			return 0;
		}

		@Override
		protected void computeColMxx(double[] c, Builtin builtin) {
		}

		@Override
		protected void computeSum(double[] c, int nRows) {
		}

		@Override
		protected void computeSumSq(double[] c, int nRows) {
		}

		@Override
		protected void computeColSumsSq(double[] c, int nRows) {
		}

		@Override
		protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		}

		@Override
		protected void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		}

		@Override
		protected void computeProduct(double[] c, int nRows) {
		}

		@Override
		protected void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		}

		@Override
		protected void computeColProduct(double[] c, int nRows) {
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
		protected void tsmm(double[] result, int numColumns, int nRows) {
		}

		@Override
		public double getIdx(int r, int colIdx) {
			return 0;
		}

		@Override
		public int getNumValues() {
			return 0;
		}

		@Override
		public CompressionType getCompType() {
			return null;
		}

		@Override
		protected ColGroupType getColGroupType() {
			return null;
		}

		@Override
		public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		}

		@Override
		public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRow) {
		}

		@Override
		public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		}

		@Override
		public AColGroup scalarOperation(ScalarOperator op) {
			return null;
		}

		@Override
		public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
			return null;
		}

		@Override
		public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
			return null;
		}

		@Override
		protected AColGroup sliceSingleColumn(int idx) {
			return null;
		}

		@Override
		public boolean containsValue(double pattern) {
			return false;
		}

		@Override
		public long getNumberNonZeros(int nRows) {
			return 0;
		}

		@Override
		public AColGroup replace(double pattern, double replace) {
			return null;
		}

		@Override
		public void computeColSums(double[] c, int nRows) {
		}

		@Override
		public CmCovObject centralMoment(CMOperator op, int nRows) {
			return null;
		}

		@Override
		public AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows) {
			return null;
		}

		@Override
		public double getCost(ComputationCostEstimator e, int nRows) {
			return 0;
		}

		@Override
		public AColGroup unaryOperation(UnaryOperator op) {
			return null;
		}

		@Override
		public AColGroup sliceRows(int rl, int ru) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AColGroup append(AColGroup g) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public ICLAScheme getCompressionScheme() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public AColGroup recompress() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'recompress'");
		}

		@Override
		public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'getCompressionInfo'");
		}

		@Override
		public boolean sameIndexStructure(AColGroupCompressed that) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'sameIndexStructure'");
		}

		@Override
		protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'fixColIndexes'");
		}

		@Override
		public void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'sparseSelection'");
		}

		@Override
		protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'denseSelection'");
		}

		@Override
		protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock dict) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToDenseBlockTransposedSparseDictionary'");
		}

		@Override
		protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToDenseBlockTransposedDenseDictionary'");
		}

		@Override
		protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock dict, int nColOut) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToSparseBlockTransposedSparseDictionary'");
		}

		@Override
		protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException(
				"Unimplemented method 'decompressToSparseBlockTransposedDenseDictionary'");
		}

		@Override
		public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'splitReshape'");
		}

		@Override
		protected boolean allowShallowIdentityRightMult() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'allowShallowIdentityRightMult'");
		}

		@Override
		public AColGroup[] splitReshapePushDown(int multiplier, int nRow, int nColOrg, ExecutorService pool)
			throws Exception {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Unimplemented method 'splitReshapePushDown'");
		}
	}
}
