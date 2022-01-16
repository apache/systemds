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

package org.apache.sysds.test.component.compress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory.CostType;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.compress.lib.CLALibAppend;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runners.Parameterized.Parameters;

public abstract class CompressedTestBase extends TestBase {
	protected static final Log LOG = LogFactory.getLog(CompressedTestBase.class.getName());

	protected static SparsityType[] usedSparsityTypes = new SparsityType[] {SparsityType.FULL, SparsityType.SPARSE,
		SparsityType.ULTRA_SPARSE};

	protected static ValueType[] usedValueTypes = new ValueType[] {ValueType.RAND_ROUND, ValueType.OLE_COMPRESSIBLE,
		ValueType.RLE_COMPRESSIBLE, ValueType.CONST};

	protected static ValueRange[] usedValueRanges = new ValueRange[] {ValueRange.BOOLEAN, ValueRange.SMALL,
		ValueRange.NEGATIVE};

	protected static OverLapping[] overLapping = new OverLapping[] {OverLapping.PLUS_LARGE, OverLapping.PLUS_ROW_VECTOR,
		OverLapping.MATRIX, OverLapping.NONE, OverLapping.APPEND_CONST, OverLapping.C_BIND_SELF};

	protected static CompressionSettingsBuilder[] usedCompressionSettings = new CompressionSettingsBuilder[] {
		// CLA TESTS!
		// csb().setValidCompressions(EnumSet.of(CompressionType.DDC)),
		// csb().setValidCompressions(EnumSet.of(CompressionType.OLE)),
		// csb().setValidCompressions(EnumSet.of(CompressionType.RLE)),
		// csb().setValidCompressions(EnumSet.of(CompressionType.SDC)),
		// csb().setValidCompressions(EnumSet.of(CompressionType.SDC, CompressionType.DDC)),

		// csb().setTransposeInput("true")
		// .setColumnPartitioner(PartitionerType.BIN_PACKING),
		// csb().setTransposeInput("true")
		// .setColumnPartitioner(PartitionerType.STATIC),

		// csb().setTransposeInput("true").setCostType(CostType.HYBRID_W_TREE),
		// csb().setTransposeInput("false"), // no transpose but default
		csb().setTransposeInput("auto").setCostType(CostType.W_TREE)};

	protected static MatrixTypology[] usedMatrixTypology = new MatrixTypology[] { // Selected Matrix Types
		MatrixTypology.SMALL, MatrixTypology.LARGE};

	private static final int compressionSeed = 7;
	protected MatrixBlock cmb;
	protected CompressionStatistics cmbStats;

	/** number of threads used for the operation */
	protected final int _k;
	protected final int sampleTolerance = 4096 * 4;

	protected double lossyTolerance;
	protected String bufferedToString;

	protected MatrixBlock vectorCols;
	protected MatrixBlock vectorRows;
	protected MatrixBlock matrixRowsCols;
	protected MatrixBlock ucRet;

	public CompressedTestBase(SparsityType sparType, ValueType valType, ValueRange valueRange,
		CompressionSettingsBuilder compSettings, MatrixTypology MatrixTypology, OverLapping ov, int parallelism,
		Collection<CompressionType> ct) {
		super(sparType, valType, valueRange, compSettings, MatrixTypology, ov, ct);

		_k = parallelism;

		CompressionSettings.PAR_DDC_THRESHOLD = 1;

		try {
			if(_cs == null && ct == null) {
				Pair<MatrixBlock, CompressionStatistics> pair = (_k == 1) ? CompressedMatrixBlockFactory
					.compress(mb) : CompressedMatrixBlockFactory.compress(mb, _k);
				cmb = pair.getLeft();
				cmbStats = pair.getRight();
			}
			else if(_cs == null) {

				cmb = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns());
				List<AColGroup> colGroups = new ArrayList<>();
				int index = 0;
				final int groupSize = (int) Math.ceil((double) mb.getNumColumns() / ct.size());
				for(CompressionType c : ct) {
					final CompressionSettings cs = csb().clearValidCompression().addValidCompression(c).create();
					final int size = Math.min(groupSize, mb.getNumColumns() - (groupSize * index));
					if(size == 0)
						continue;
					final int[] colIndexes = new int[Math.min(groupSize, mb.getNumColumns() - (groupSize * index))];
					for(int x = 0; x < colIndexes.length; x++) {
						int y = index * groupSize + x;
						colIndexes[x] = y;
					}

					ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, mb, false, 8, true);
					EstimationFactors ef = CompressedSizeEstimator.estimateCompressedColGroupSize(ubm, colIndexes,
						mb.getNumRows(), cs);
					CompressedSizeInfoColGroup cgi = new CompressedSizeInfoColGroup(colIndexes, ef, c);
					CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
					for(AColGroup cg : ColGroupFactory.compressColGroups(mb, csi, cs, 1))
						colGroups.add(cg);
					index++;
				}
				((CompressedMatrixBlock) cmb).allocateColGroupList(colGroups);
				cmb.recomputeNonZeros();
			}
			else {
				if(_cs != null && (_cs.lossy || ov == OverLapping.SQUASH))
					setLossyTolerance(valueRange);

				if(_cs.validCompressions.size() == 2) {
					/**
					 * In case only Uncompressed and Const colgroups are available. filter the big tests from uncompressed
					 * colgroup tests since the functionality should be verified even with smaller matrices
					 */

					if(rows < 10000) {
						int[] colIndexes = new int[mb.getNumColumns()];
						for(int i = 0; i < colIndexes.length; i++)
							colIndexes[i] = i;
						cmb = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns());
						((CompressedMatrixBlock) cmb).allocateColGroup(new ColGroupUncompressed(colIndexes, mb, false));
					}
				}
				else {
					Pair<MatrixBlock, CompressionStatistics> pair = CompressedMatrixBlockFactory.compress(mb, _k, _csb);
					cmb = pair.getLeft();
					cmbStats = pair.getRight();
				}
			}
			MatrixBlock tmp = null;
			switch(ov) {
				case COL:
					tmp = TestUtils.generateTestMatrixBlock(cols, 1, 0.5, 1.5, 1.0, 6);
					lossyTolerance = lossyTolerance * 80;
					cols = 1;
					break;
				case MATRIX:
				case MATRIX_MULT_NEGATIVE:
				case MATRIX_PLUS:
				case SQUASH:
					tmp = TestUtils.round(TestUtils.generateTestMatrixBlock(cols, 2, 0, 5, 1.0, 2));
					lossyTolerance = lossyTolerance * 160;
					cols = 2;
					break;
				case APPEND_EMPTY:
					tmp = new MatrixBlock(rows, 1, 0);
					break;
				case APPEND_CONST:
					tmp = new MatrixBlock(rows, 1, 0).scalarOperations(new RightScalarOperator(Plus.getPlusFnObject(), 1),
						new MatrixBlock());
					break;
				case C_BIND_SELF:
					if(cmb instanceof CompressedMatrixBlock) {
						CompressedMatrixBlock cmbc = (CompressedMatrixBlock) cmb;
						cmb = CLALibAppend.append(cmbc, cmbc, false);
						mb = mb.append(mb, new MatrixBlock());
						cols *= 2;
					}
					break;
				default:
					break;
			}
			if(cmb instanceof CompressedMatrixBlock) {
				if(tmp != null && ov == OverLapping.APPEND_EMPTY || ov == OverLapping.APPEND_CONST) {
					mb = mb.append(tmp, new MatrixBlock());
					cmb = cmb.append(tmp, new MatrixBlock());
					cols += tmp.getNumColumns();
				}
				else if(tmp != null) {
					// Make Operator
					AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

					mb = mb.aggregateBinaryOperations(mb, tmp, new MatrixBlock(), abop);
					cmb = cmb.aggregateBinaryOperations(cmb, tmp, new MatrixBlock(), abop);

					if(ov == OverLapping.MATRIX_PLUS) {
						ScalarOperator sop = new LeftScalarOperator(Plus.getPlusFnObject(), 15);
						mb = mb.scalarOperations(sop, new MatrixBlock());
						cmb = cmb.scalarOperations(sop, new MatrixBlock());
					}
					else if(ov == OverLapping.MATRIX_MULT_NEGATIVE) {
						ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), -1.3);
						mb = mb.scalarOperations(sop, new MatrixBlock());
						cmb = cmb.scalarOperations(sop, new MatrixBlock());
					}
					else if(ov == OverLapping.SQUASH) {
						if(cmb instanceof CompressedMatrixBlock)
							cmb = ((CompressedMatrixBlock) cmb).squash(_k);
					}
				}
				if(cmb instanceof CompressedMatrixBlock) {

					if(ov == OverLapping.PLUS || ov == OverLapping.PLUS_LARGE) {
						ScalarOperator sop = ov == OverLapping.PLUS_LARGE ? new LeftScalarOperator(Plus.getPlusFnObject(),
							-3142151) : new LeftScalarOperator(Plus.getPlusFnObject(), 5);
						mb = mb.scalarOperations(sop, new MatrixBlock());
						cmb = cmb.scalarOperations(sop, new MatrixBlock());
					}
					else if(ov == OverLapping.PLUS_ROW_VECTOR) {

						MatrixBlock v = TestUtils.generateTestMatrixBlock(1, cols, -1, 1, 1.0, 4);
						BinaryOperator bop = new BinaryOperator(Plus.getPlusFnObject(), _k);
						mb = mb.binaryOperations(bop, v, null);
						cmb = cmb.binaryOperations(bop, v, null);
						lossyTolerance = lossyTolerance + 2;
					}
					if(!(cmb instanceof CompressedMatrixBlock))
						fail("Invalid construction, should result in compressed MatrixBlock");
				}
			}

			bufferedToString = this.toString();
			if(cmb instanceof CompressedMatrixBlock) {

				vectorCols = TestUtils.generateTestMatrixBlock(1, cols, -1.0, 1.5, 1.0, 3);
				vectorRows = CompressibleInputGenerator.getInput(rows, 1, CompressionType.OLE, 5, 5, -5, 1.0, 3);
				matrixRowsCols = TestUtils.generateTestMatrixBlock(rows, cols, -1.0, 1.5, 1.0, 3);
			}
			else {
				vectorCols = null;
				vectorRows = null;
				matrixRowsCols = null;
			}
			TestUtils.assertEqualColsAndRows(mb, cmb, bufferedToString);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("\nCompressionTest Init failed with settings: " + this.toString());
		}
	}

	/**
	 * Tolerance for encoding values is the maximum value in dataset divided by number distinct values available in a
	 * single Byte (since we encode our quntization in Byte)
	 * 
	 * @param valueRange The value range used as input
	 */
	private void setLossyTolerance(ValueRange valueRange) {
		lossyTolerance = (double) (Math.max(TestConstants.getMaxRangeValue(valueRange),
			Math.abs(TestConstants.getMinRangeValue(valueRange)))) * (1.0 / 127.0) / 2.0;
	}

	@Parameters
	public static Collection<Object[]> data() {
		ArrayList<Object[]> tests = new ArrayList<>();

		for(SparsityType st : usedSparsityTypes)
			for(ValueType vt : usedValueTypes)
				for(ValueRange vr : usedValueRanges)
					for(CompressionSettingsBuilder cs : usedCompressionSettings)
						for(MatrixTypology mt : usedMatrixTypology)
							for(OverLapping ov : overLapping)
								if((ov == OverLapping.APPEND_CONST || ov == OverLapping.APPEND_EMPTY)) {
									if(vr == ValueRange.BOOLEAN)
										tests.add(new Object[] {st, vt, vr, cs, mt, ov, null});
								}
								else
									tests.add(new Object[] {st, vt, vr, cs, mt, ov, null});

		final MatrixTypology mt = MatrixTypology.SMALL;
		for(CompressionSettingsBuilder cs : usedCompressionSettings) {
			for(OverLapping ov : overLapping) {
				tests.add(new Object[] {SparsityType.EMPTY, ValueType.RAND, ValueRange.BOOLEAN, cs, mt, ov, null});
				tests.add(new Object[] {SparsityType.FULL, ValueType.CONST, ValueRange.LARGE, cs, mt, ov, null});
				tests.add(new Object[] {SparsityType.FULL, ValueType.ONE_HOT, ValueRange.BOOLEAN, cs, mt, ov, null});
			}
		}
		final CompressionSettingsBuilder cs = csb().setColumnPartitioner(PartitionerType.STATIC)
			.setTransposeInput("true");

		CompressionType[] forcedColGroups = new CompressionType[] {CompressionType.DDC, CompressionType.SDC,
			CompressionType.UNCOMPRESSED};
		ValueType[] forcedCompressionValueTypes = new ValueType[] {ValueType.OLE_COMPRESSIBLE, ValueType.CONST};
		for(ValueType vt : forcedCompressionValueTypes) {
			SparsityType st = SparsityType.SPARSE;
			for(OverLapping ov : overLapping) {
				tests.add(new Object[] {st, vt, ValueRange.BOOLEAN, null, mt, ov, null});
				List<CompressionType> ctl = Arrays.asList(forcedColGroups);
				tests.add(new Object[] {st, vt, ValueRange.SMALL, null, mt, ov, ctl});
			}
		}

		// add special cases
		final OverLapping ov = OverLapping.NONE;
		final OverLapping cBindSelf = OverLapping.C_BIND_SELF;
		final SparsityType sp = SparsityType.ULTRA_SPARSE;
		final ValueType ubs = ValueType.UNBALANCED_SPARSE;

		tests.add(new Object[] {sp, ValueType.OLE_COMPRESSIBLE, ValueRange.CONST, cs, mt, ov, null});
		tests.add(new Object[] {sp, ValueType.OLE_COMPRESSIBLE, ValueRange.SMALL, cs, mt, ov, null});
		tests.add(new Object[] {sp, ubs, ValueRange.CONST, cs, mt, ov, null});
		tests.add(new Object[] {sp, ubs, ValueRange.SMALL, cs, mt, ov, null});

		final ValueType rd = ValueType.RAND;

		final CompressionType uc = CompressionType.UNCOMPRESSED;
		List<CompressionType> forceUncompressed = Arrays.asList(new CompressionType[] {uc, uc});
		// forced two uncompressed
		tests.add(new Object[] {sp, rd, ValueRange.SMALL, null, mt, ov, forceUncompressed});
		tests.add(new Object[] {sp, rd, ValueRange.SMALL, null, mt, cBindSelf, forceUncompressed});
		// Ubs to ensure that one of the compressed matrices is empty.
		tests.add(new Object[] {sp, ubs, ValueRange.SMALL, null, mt, cBindSelf, forceUncompressed});
		// forced two uncompressed empty colGroups.
		tests.add(new Object[] {SparsityType.EMPTY, rd, ValueRange.CONST, null, mt, ov, forceUncompressed});

		return tests;

	}

	@Test
	public void testDecompress() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			((CompressedMatrixBlock) cmb).clearSoftReferenceToDecompressed();
			MatrixBlock decompressedMatrixBlock = ((CompressedMatrixBlock) cmb).decompress(_k);
			compareResultMatrices(mb, decompressedMatrixBlock, 1);
			assertEquals(bufferedToString, mb.getNonZeros(), decompressedMatrixBlock.getNonZeros());
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixMultChainXtXv() {
		testMatrixMultChain(ChainType.XtXv);
	}

	@Test
	public void testMatrixMultChainXtwXv() {
		testMatrixMultChain(ChainType.XtwXv);
	}

	@Test
	@Ignore
	public void testMatrixMultChainXtXvy() {
		testMatrixMultChain(ChainType.XtXvy);
	}

	public void testMatrixMultChain(ChainType ctype) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector1 = TestUtils.generateTestMatrixBlock(cols, 1, 0.9, 1.5, 1.0, 3);

			MatrixBlock vector2 = (ctype == ChainType.XtwXv) ? vectorRows : null;

			// matrix-vector uncompressed
			ucRet = mb.chainMatrixMultOperations(vector1, vector2, ucRet, ctype, _k);

			// matrix-vector compressed
			MatrixBlock ret2 = cmb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, _k);

			// compare result with input
			compareResultMatrices(ucRet, ret2, 4);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testVectorMatrixMult() {
		MatrixBlock vector = TestUtils.generateTestMatrixBlock(1, rows, 0.9, 1.5, 1.0, 3);
		testLeftMatrixMatrix(vector);
	}

	@Test
	public void testLeftMatrixMatrixMultSmall() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(3, rows, 0.9, 1.5, 1.0, 3);
		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testLeftMatrixMatrixMultSparse() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(2, rows, 0.9, 1.5, .1, 3);
		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testLeftMatrixMatrixMultSparseCustom() {
		MatrixBlock matrix = new MatrixBlock(2, rows, true);
		matrix.quickSetValue(1, rows - 1, 99);
		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testLeftMatrixMatrixMultSparseCustom2() {
		MatrixBlock matrix = new MatrixBlock(2, rows, true);
		matrix.quickSetValue(1, 0, 99);
		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testLeftMatrixMatrixMultSparseCustom3() {
		MatrixBlock matrix = new MatrixBlock(2, rows, true);
		matrix.quickSetValue(0, 0, -99);
		matrix.quickSetValue(1, 0, 99);
		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testLeftMatrixMatrixMultSparseCustom4() {
		MatrixBlock matrix = new MatrixBlock(2, rows, true);
		matrix.quickSetValue(0, rows - 1, -99);
		matrix.quickSetValue(1, 0, 99);
		testLeftMatrixMatrix(matrix);
	}

	public void testLeftMatrixMatrix(MatrixBlock matrix) {
		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test
		try {
			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			ucRet = mb.aggregateBinaryOperations(matrix, mb, ucRet, abop);
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(matrix, cmb, new MatrixBlock(), abop);

			// compare result with input
			compareResultMatrices(ucRet, ret2, 100);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixVectorMult03() {
		testMatrixVectorMult(-1.0, 1.0);
	}

	public void testMatrixVectorMult(double min, double max) {
		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock vector = TestUtils.generateTestMatrixBlock(cols, 1, min, max, 1.0, 3);
		testRightMatrixMatrix(vector);
	}

	@Test
	public void testRightMatrixMatrixMultSmall() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(cols, 2, 0.9, 1.5, 1.0, 3);
		testRightMatrixMatrix(matrix);
	}

	@Test
	@Ignore
	public void testRightMatrixMatrixMultMedium() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(cols, 16, 0.9, 1.5, 1.0, 3);
		testRightMatrixMatrix(matrix);
	}

	@Test
	public void testRightMatrixMatrixMultSparse() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(cols, 25, 0.9, 1.5, 0.01, 3);
		testRightMatrixMatrix(matrix);
	}

	public void testRightMatrixMatrix(MatrixBlock matrix) {
		try {
			matrix.quickSetValue(0, 0, 10);
			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// vector-matrix uncompressed
			ucRet = mb.aggregateBinaryOperations(mb, matrix, ucRet, abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(cmb, matrix, new MatrixBlock(), abop);

			// compare result with input
			compareResultMatrices(ucRet, ret2, 10);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testLeftMatrixMatrixMultTransposedLeftSide() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(rows, 2, 0.9, 1.5, 1.0, 3);
		testLeftMatrixMatrixMultiplicationTransposed(matrix, true, false, false, true);
	}

	@Test
	public void testLeftMatrixMatrixMultTransposedRightSide() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(2, cols, 0.9, 1.5, 1.0, 3);
		testLeftMatrixMatrixMultiplicationTransposed(matrix, false, true, false, false);
	}

	@Test
	public void testLeftMatrixMatrixMultTransposedBothSides() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(cols, 1, 0.9, 1.5, 1.0, 3);
		testLeftMatrixMatrixMultiplicationTransposed(matrix, true, true, false, false);
	}

	@Test
	public void testLeftMatrixMatrixMultDoubleCompressedTransposedLeftSideSmaller() {
		MatrixBlock matrix = CompressibleInputGenerator.getInput(rows, 1, CompressionType.OLE, 5, 20, -20, 1.0, 3);
		testLeftMatrixMatrixMultiplicationTransposed(matrix, true, false, true, true);
	}

	@Test
	public void testLeftMatrixMatrixMultDoubleCompressedTransposedLeftSideBigger() {
		MatrixBlock matrix = CompressibleInputGenerator.getInput(rows, cols + 1, CompressionType.OLE, 5, 20, -20, 1.0, 3);
		testLeftMatrixMatrixMultiplicationTransposed(matrix, true, false, true, true);
	}

	@Test
	@Ignore
	public void testLeftMatrixMatrixMultDoubleCompressedTransposedBothSides() {
		// This test does not currently work, since the intension is that the "transposed" input matrix is compressed and
		// not transposed.
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(1, rows, -20, 20, 1.0, 3);
		testLeftMatrixMatrixMultiplicationTransposed(matrix, true, true, true, true);
	}

	public void testLeftMatrixMatrixMultiplicationTransposed(MatrixBlock matrix, boolean transposeLeft,
		boolean transposeRight, boolean compressMatrix, boolean comparePercent) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// vector-matrix compressed
			CompressionSettingsBuilder cs = new CompressionSettingsBuilder()
				.setValidCompressions(EnumSet.of(CompressionType.DDC));
			MatrixBlock compMatrix = compressMatrix ? CompressedMatrixBlockFactory.compress(matrix, cs).getLeft() : matrix;
			if(compressMatrix && !(compMatrix instanceof CompressedMatrixBlock))
				return; // Early termination since the test does not test what we wanted.

			// Make Operator
			AggregateBinaryOperator abopSingle = InstructionUtils.getMatMultOperator(1);

			// vector-matrix uncompressed
			ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _k);

			// vector-matrix compressed
			MatrixBlock left = transposeLeft ? matrix.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0) : matrix;
			MatrixBlock right = transposeRight ? mb.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0) : mb;
			ucRet = right.aggregateBinaryOperations(left, right, ucRet, abopSingle);

			MatrixBlock ret2 = ((CompressedMatrixBlock) cmb).aggregateBinaryOperations(compMatrix, cmb, new MatrixBlock(),
			abopSingle, transposeLeft, transposeRight);

			compareResultMatrices(ucRet, ret2, 100);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testRightMatrixMatrixMultTransposedLeftSide() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(rows, 2, 0.9, 1.5, 1.0, 3);
		testRightMatrixMatrixMultiplicationTransposed(matrix, true, false, false);
	}

	@Test
	public void testRightMatrixMatrixMultTransposedRightSide() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(3, cols, 0.9, 1.5, 1.0, 3);
		testRightMatrixMatrixMultiplicationTransposed(matrix, false, true, false);
	}

	@Test
	public void testRightMatrixMatrixMultTransposedBothSides() {
		MatrixBlock matrix = TestUtils.generateTestMatrixBlock(1, rows, 0.9, 1.5, 1.0, 3);
		testRightMatrixMatrixMultiplicationTransposed(matrix, true, true, false);
	}

	public void testRightMatrixMatrixMultiplicationTransposed(MatrixBlock matrix, boolean transposeLeft,
		boolean transposeRight, boolean compressMatrix) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// vector-matrix uncompressed
			ReorgOperator r_op = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _k);

			MatrixBlock left = transposeLeft ? mb.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0) : mb;
			MatrixBlock right = transposeRight ? matrix.reorgOperations(r_op, new MatrixBlock(), 0, 0, 0) : matrix;
			ucRet = right.aggregateBinaryOperations(left, right, ucRet, abop);

			// vector-matrix compressed
			MatrixBlock compMatrix = compressMatrix ? CompressedMatrixBlockFactory.compress(matrix, _k).getLeft() : matrix;
			MatrixBlock ret2 = ((CompressedMatrixBlock) cmb).aggregateBinaryOperations(cmb, compMatrix, new MatrixBlock(),
				abop, transposeLeft, transposeRight);

			compareResultMatrices(ucRet, ret2, 100);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testLeftMultWithSlightlyModifiedSelf() {
		try {
			// This test is for matrix multiplication where the matrix is slightly modified and then transposed and
			// multiplied on the left.
			if(!(cmb instanceof CompressedMatrixBlock) && rows > 600)
				return; // Input was not compressed then just pass test
			final ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), 3, _k);
			MatrixBlock cmbM = cmb.scalarOperations(sop, null);
			if(!(cmbM instanceof CompressedMatrixBlock))
				return; // the modified version was not compressed
			final AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// Modify both the compressed and uncompressed matrix.
			MatrixBlock mbMt = LibMatrixReorg.transpose(mb.scalarOperations(sop, ucRet), _k);

			// matrix-vector compressed
			MatrixBlock ret2 = ((CompressedMatrixBlock) cmb).aggregateBinaryOperations(cmbM, cmb, null, abop, true, false);
			// matrix-vector uncompressed
			ucRet = mb.aggregateBinaryOperations(mbMt, mb, ucRet, abop);

			// compare result with input
			compareResultMatrices(ucRet, ret2, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testTransposeSelfMatrixMultLeft() {
		// TSMM tsmm
		try {
			testTransposeSelfMatrixMult(MMTSJType.LEFT);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void testTransposeSelfMatrixMultRight() {
		if(!(cmb instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Make test pass");
		testTransposeSelfWithExpectedException(MMTSJType.RIGHT);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testTransposeSelfMatrixMultNONE() {
		if(!(cmb instanceof CompressedMatrixBlock))
			throw new DMLRuntimeException("Make test pass");
		testTransposeSelfWithExpectedException(MMTSJType.NONE);
	}

	public void testTransposeSelfMatrixMult(MMTSJType mType) {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		if(_k != 1) {
			// matrix-vector compressed
			MatrixBlock ret2 = cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, _k);
			// matrix-vector uncompressed
			ucRet = mb.transposeSelfMatrixMultOperations(ucRet, mType, _k);
			// compare result with input
			compareResultMatrices(ucRet, ret2, 1);
		}
		else {
			// matrix-vector compressed
			MatrixBlock ret2 = cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType);
			// matrix-vector uncompressed
			ucRet = mb.transposeSelfMatrixMultOperations(ucRet, mType);
			// compare result with input
			compareResultMatrices(ucRet, ret2, 1);
		}
	}

	public void testTransposeSelfWithExpectedException(MMTSJType mType) {
		if(_k != 1)
			cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, _k);
		else
			cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType);
	}

	@Test
	public void testScalarOpRightMultiplyPositive() {
		double mult = 7;
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarOpRightDivide() {
		double mult = 0.2;
		ScalarOperator sop = new RightScalarOperator(Divide.getDivideFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarOpRightMultiplyNegative() {
		double mult = -7;
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarRightOpAddition() {
		double addValue = 4;
		ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.05);
	}

	@Test
	public void testScalarLeftOpGreater() {
		double addValue = 0.11;
		ScalarOperator sop = new LeftScalarOperator(GreaterThan.getGreaterThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	public void testScalarOperations(ScalarOperator sop, double tolerance) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			ucRet = mb.scalarOperations(sop, ucRet);
			// matrix-scalar compressed
			MatrixBlock ret2 = cmb.scalarOperations(sop, new MatrixBlock());

			// compare result with input
			compareResultMatrices(ucRet, ret2, tolerance);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testBinaryMVAdditionROW() {
		ValueFunction vf = Plus.getPlusFnObject();
		testBinaryMV(vf, vectorCols);
	}

	@Test
	public void testBinaryMVAdditionCOL() {
		ValueFunction vf = Plus.getPlusFnObject();
		testBinaryMV(vf, vectorRows);
	}

	@Test
	public void testBinaryMVMultiplyROW() {
		ValueFunction vf = Multiply.getMultiplyFnObject();
		testBinaryMV(vf, vectorCols);
	}

	@Test
	public void testBinaryMVDivideROW() {
		ValueFunction vf = Divide.getDivideFnObject();
		testBinaryMV(vf, vectorCols);
	}

	@Test
	public void testBinaryVMPlusRow() {
		testBinaryVMPlus(vectorCols);
	}

	@Test
	public void testBinaryVMPlusCols() {
		testBinaryVMPlus(vectorRows);
	}

	private static AtomicBoolean printedErrorForNotImplementedTestBinaryVMPlus = new AtomicBoolean(false);

	public void testBinaryVMPlus(MatrixBlock vector) {
		// This test verifies that left binary operations work. but they are not integrated into the system
		// Since this operation is not supported in normal MatrixBlock.
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				// (rows * cols > 10000 && matrix.getNumRows() == rows && matrix.getNumColumns() == cols))
				return; // Input was not compressed then just pass test
			for(AColGroup g : ((CompressedMatrixBlock) cmb).getColGroups())
				if(g instanceof ColGroupUncompressed)
					return; // Not supported for colGroupUncompressed.
			BinaryOperator bop = new BinaryOperator(Plus.getPlusFnObject(), _k);
			MatrixBlock matrix = vector;
			MatrixBlock ret2;
			ucRet = mb.binaryOperations(bop, matrix, ucRet);
			ret2 = ((CompressedMatrixBlock) cmb).binaryOperationsLeft(bop, matrix, new MatrixBlock());

			compareResultMatrices(ucRet, ret2, 2);
		}
		catch(NotImplementedException e) {
			if(! printedErrorForNotImplementedTestBinaryVMPlus.get()){
				LOG.error("Failed Binary VM Plus: " + e.getMessage());
				printedErrorForNotImplementedTestBinaryVMPlus.set(true);
			}
		}
		catch(Exception e) {
			if(e.getCause() instanceof ExecutionException && e.getCause().getCause() instanceof NotImplementedException) {
				if(! printedErrorForNotImplementedTestBinaryVMPlus.get()){
					LOG.error("Failed Binary VM Plus: " + e.getMessage());
					printedErrorForNotImplementedTestBinaryVMPlus.set(true);
				}
			}
			else {
				e.printStackTrace();
				fail("Not correct error message" + e.getMessage());
			}
		}
	}


	public void testBinaryMV(ValueFunction vf, MatrixBlock matrix) {
		testBinaryMV(vf, matrix, true);
	}

	public void testBinaryMV(ValueFunction vf, MatrixBlock matrix, boolean right) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				// (rows * cols > 10000 && matrix.getNumRows() == rows && matrix.getNumColumns() == cols))
				return; // Input was not compressed then just pass test

			BinaryOperator bop = new BinaryOperator(vf, _k);
			MatrixBlock ret2;
			if(right) {
				ucRet = mb.binaryOperations(bop, matrix, ucRet);
				ret2 = cmb.binaryOperations(bop, matrix, new MatrixBlock());
			}
			else {
				ucRet = matrix.binaryOperations(bop, mb, ucRet);
				ret2 = matrix.binaryOperations(bop, cmb, new MatrixBlock());
			}
			compareResultMatrices(ucRet, ret2, 2);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testSliceRows() {
		testSlice(rows / 5, Math.min(rows - 1, (rows / 5) * 2), 0, cols - 1);
	}

	@Test
	public void testSliceFirstColumn() {
		testSlice(0, rows - 1, 0, 0);
	}

	@Test
	public void testSliceLastColumn() {
		testSlice(0, rows - 1, cols - 1, cols - 1);
	}

	@Test
	public void testSliceAllButFirstColumn() {
		testSlice(0, rows - 1, Math.min(1, cols - 1), cols - 1);
	}

	@Test
	public void testSliceInternal() {
		testSlice(rows / 5, Math.min(rows - 1, (rows / 5) * 2), Math.min(cols - 1, cols / 5),
			Math.min(cols - 1, cols / 5 + 1));
	}

	@Test
	public void testSliceFirstValue() {
		testSlice(0, 0, 0, 0);
	}

	@Test
	public void testSliceEntireMatrix() {
		testSlice(0, rows - 1, 0, cols - 1);
	}

	public void testSlice(int rl, int ru, int cl, int cu) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			final MatrixBlock ret2 = cmb.slice(rl, ru, cl, cu);
			final MatrixBlock ret1 = mb.slice(rl, ru, cl, cu);
			final long nnz1 = ret1.getNonZeros();
			final long nnz2 = ret2.getNonZeros();
			if(!(ret2 instanceof CompressedMatrixBlock) && nnz1 != nnz2)
				fail(bufferedToString + "\nNot same number of non zeros " + nnz1 + " != " + nnz2);

			compareResultMatrices(ret1, ret2, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Error in Slicing", e);
		}
	}

	@Test
	public void testCompressAgain() {
		try {
			TestUtils.assertEqualColsAndRows(mb, cmb);
			MatrixBlock cmba = CompressedMatrixBlockFactory.compress(cmb, _k).getLeft();
			compareResultMatrices(mb, cmba, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void append() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		MatrixBlock ap = vectorRows;
		MatrixBlock ret1 = mb.append(ap, new MatrixBlock());
		MatrixBlock ret2 = cmb.append(ap, new MatrixBlock());
		compareResultMatrices(ret1, ret2, 1);
	}

	@Test
	public void appendCBindTrue() {
		if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
			return;
		MatrixBlock ap = new MatrixBlock(mb.getNumRows(), 1, false, 132);
		MatrixBlock ret1 = mb.append(ap, new MatrixBlock(), true);
		MatrixBlock ret2 = cmb.append(ap, new MatrixBlock(), true);
		compareResultMatrices(ret1, ret2, 1);
	}

	@Test
	public void appendCBindFalse() {
		if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
			return;
		MatrixBlock ap = new MatrixBlock(1, mb.getNumColumns(), false, 132);
		MatrixBlock ret1 = mb.append(ap, new MatrixBlock(), false);
		MatrixBlock ret2 = cmb.append(ap, new MatrixBlock(), false);
		compareResultMatrices(ret1, ret2, 1);
	}

	@Test
	public void unaryRoundTest() {
		unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ROUND)));
	}

	@Test
	public void unaryIsNANTest() {
		unaryOperations(new UnaryOperator(Builtin.getBuiltinFnObject(BuiltinCode.ISNAN)));
	}

	public void unaryOperations(UnaryOperator op) {
		if(!(cmb instanceof CompressedMatrixBlock))
			return;
		try {
			ucRet = mb.unaryOperations(op, ucRet);
			MatrixBlock ret2 = cmb.unaryOperations(op, null);
			compareResultMatrices(ucRet, ret2, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testRandOperationsInPlace() {
		if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
			return;
		final double min = -100;
		final double max = 100;
		final double sparsity = 0.3;
		RandomMatrixGenerator rgen = new RandomMatrixGenerator("uniform", rows, cols, ConfigurationManager.getBlocksize(),
			sparsity, min, max);
		if(!LibMatrixDatagen.isShortcutRandOperation(min, max, sparsity, RandomMatrixGenerator.PDF.UNIFORM)) {
			MatrixBlock ret1 = cmb.randOperationsInPlace(rgen, LibMatrixDatagen.setupSeedsForRand(seed), 1342);
			MatrixBlock ret2 = mb.randOperationsInPlace(rgen, LibMatrixDatagen.setupSeedsForRand(seed), 1342);
			compareResultMatrices(ret1, ret2, 1);
		}
		else
			fail("Invalid random setup for test.");
	}

	protected void compareResultMatrices(MatrixBlock expected, MatrixBlock result, double toleranceMultiplier) {
		TestUtils.assertEqualColsAndRows(expected, result);
		if(expected instanceof CompressedMatrixBlock)
			expected = ((CompressedMatrixBlock) expected).decompress();
		if(result instanceof CompressedMatrixBlock)
			result = ((CompressedMatrixBlock) result).decompress();

		if(_cs != null && _cs.lossy)
			TestUtils.compareMatricesPercentageDistance(expected, result, 0.25, 0.83, bufferedToString);
		else if(overlappingType == OverLapping.SQUASH)
			TestUtils.compareMatrices(expected, result, lossyTolerance * toleranceMultiplier * 1.3, bufferedToString);
		// else if(rows > 65000)
		// TestUtils.compareMatricesPercentageDistance(expected, result, 0.99, 0.99, bufferedToString);
		else if(OverLapping.effectOnOutput(overlappingType))
			TestUtils.compareMatricesPercentageDistance(expected, result, 0.99, 0.99, bufferedToString);
		else
			TestUtils.compareMatricesBitAvgDistance(expected, result, (long) (27000 * toleranceMultiplier),
				(long) (1024 * toleranceMultiplier), bufferedToString);
	}

	protected static CompressionSettingsBuilder csb() {
		return new CompressionSettingsBuilder().setSeed(compressionSeed).setMinimumSampleSize(100);
	}
}
