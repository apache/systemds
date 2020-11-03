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

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Xor;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;
import org.junit.runners.Parameterized.Parameters;

public abstract class CompressedTestBase extends TestBase {
	protected static final Log LOG = LogFactory.getLog(CompressedTestBase.class.getName());

	protected static SparsityType[] usedSparsityTypes = new SparsityType[] { // Sparsity 0.9, 0.1, 0.01 and 0.0
		// SparsityType.FULL,
		SparsityType.DENSE,
		// SparsityType.SPARSE,
		SparsityType.ULTRA_SPARSE,
		// SparsityType.EMPTY
	};

	protected static ValueType[] usedValueTypes = new ValueType[] {
		// ValueType.RAND,
		// ValueType.CONST,
		ValueType.RAND_ROUND, ValueType.OLE_COMPRESSIBLE,
		// ValueType.RLE_COMPRESSIBLE,
	};

	protected static ValueRange[] usedValueRanges = new ValueRange[] {ValueRange.SMALL,
		// ValueRange.LARGE,
		// ValueRange.BYTE
	};

	protected static OverLapping[] overLapping = new OverLapping[] {OverLapping.COL, OverLapping.MATRIX,
		OverLapping.NONE, OverLapping.MATRIX_PLUS, OverLapping.MATRIX_MULT_NEGATIVE};

	private static final int compressionSeed = 7;

	protected static CompressionSettings[] usedCompressionSettings = new CompressionSettings[] {
		// CLA TESTS!

		new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed)
			.setValidCompressions(EnumSet.of(CompressionType.DDC)).setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed)
			.setValidCompressions(EnumSet.of(CompressionType.OLE)).setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed)
			.setValidCompressions(EnumSet.of(CompressionType.RLE)).setInvestigateEstimate(true).create(),
		new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed).setInvestigateEstimate(true)
			.create(),
		new CompressionSettingsBuilder().setSamplingRatio(1.0).setSeed(compressionSeed).setInvestigateEstimate(true)
			.setAllowSharedDictionary(false).setmaxStaticColGroupCoCode(1).create(),

		// // // // LOSSY TESTS!

		// new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed)
		// .setValidCompressions(EnumSet.of(CompressionType.DDC)).setInvestigateEstimate(true).setLossy(true).create(),
		// new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed)
		// .setValidCompressions(EnumSet.of(CompressionType.OLE)).setInvestigateEstimate(true).setLossy(true).create(),
		// new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed)
		// .setValidCompressions(EnumSet.of(CompressionType.RLE)).setInvestigateEstimate(true).setLossy(true).create(),
		// new CompressionSettingsBuilder().setSamplingRatio(0.1).setSeed(compressionSeed).setInvestigateEstimate(true)
		// .create(),
		// new CompressionSettingsBuilder().setSamplingRatio(1.0).setSeed(compressionSeed).setInvestigateEstimate(true)
		// .setAllowSharedDictionary(false).setmaxStaticColGroupCoCode(1).setLossy(true).create(),

		// COCODING TESTS!!

		// new CompressionSettingsBuilder().setSamplingRatio(1.0).setSeed(compressionSeed).setInvestigateEstimate(true)
		// .setAllowSharedDDCDictionary(false).setmaxStaticColGroupCoCode(20).create(),
		// new CompressionSettingsBuilder().setSamplingRatio(1.0).setSeed(compressionSeed).setInvestigateEstimate(true)
		// .setAllowSharedDDCDictionary(false).setmaxStaticColGroupCoCode(20).setLossy(true).create()

		// SHARED DICTIONARY TESTS!!

	};

	protected static MatrixTypology[] usedMatrixTypology = new MatrixTypology[] { // Selected Matrix Types
		// MatrixTypology.SMALL, MatrixTypology.FEW_COL,
		// MatrixTypology.FEW_ROW,
		// MatrixTypology.LARGE,
		// MatrixTypology.SINGLE_COL,
		// MatrixTypology.SINGLE_ROW,
		MatrixTypology.L_ROWS,
		// MatrixTypology.XL_ROWS,
		// MatrixTypology.SINGLE_COL_L
	};

	// Compressed Block
	protected MatrixBlock cmb;
	protected CompressionStatistics cmbStats;

	// Decompressed Result
	// protected MatrixBlock cmbDeCompressed;
	// protected double[][] deCompressed;

	/** number of threads used for the operation */
	protected final int _k;

	protected int sampleTolerance = 4096 * 4;

	protected double lossyTolerance;

	public CompressedTestBase(SparsityType sparType, ValueType valType, ValueRange valueRange,
		CompressionSettings compSettings, MatrixTypology MatrixTypology, OverLapping ov, int parallelism) {
		super(sparType, valType, valueRange, compSettings, MatrixTypology, ov);

		_k = parallelism;

		try {
			if(compSettings.lossy)
				setLossyTolerance(valueRange);
			Pair<MatrixBlock, CompressionStatistics> pair = CompressedMatrixBlockFactory
				.compress(mb, _k, compressionSettings);
			cmb = pair.getLeft();
			cmbStats = pair.getRight();
			MatrixBlock tmp = null;
			switch(ov) {
				case COL:
					tmp = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 0.5, 1.5, 1.0, 6));
					lossyTolerance = lossyTolerance * 80;
					cols = 1;
					break;
				case MATRIX:
				case MATRIX_MULT_NEGATIVE:
				case MATRIX_PLUS:
					tmp = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 2, 0.5, 1.5, 1.0, 2));
					lossyTolerance = lossyTolerance * 160;
					cols = 2;
					break;
				default:
					break;
			}
			if(cmb instanceof CompressedMatrixBlock) {
				if(tmp != null) {
					// Make Operator
					AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

					// vector-matrix uncompressed
					mb = mb.aggregateBinaryOperations(mb, tmp, new MatrixBlock(), abop);

					// vector-matrix compressed
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
				}
			}

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
					for(CompressionSettings cs : usedCompressionSettings)
						for(MatrixTypology mt : usedMatrixTypology)
							for(OverLapping ov : overLapping)
								tests.add(new Object[] {st, vt, vr, cs, mt, ov});
		return tests;
	}

	@Test
	public void testDecompress() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock)) {
				return; // Input was not compressed then just pass test
				// Assert.assertTrue("Compression Failed \n" + this.toString(), false);
			}
			double[][] org = DataConverter.convertToDoubleMatrix(mb);
			double[][] deCompressed = DataConverter.convertToDoubleMatrix(((CompressedMatrixBlock) cmb).decompress(_k));
			if(compressionSettings.lossy)
				TestUtils.compareMatrices(org, deCompressed, lossyTolerance, this.toString());
			else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE || overlappingType == OverLapping.MATRIX_PLUS ||
				overlappingType == OverLapping.MATRIX || overlappingType == OverLapping.COL)
				TestUtils.compareMatricesBitAvgDistance(org, deCompressed, 8192, 124, this.toString());
			else
				TestUtils.compareMatricesBitAvgDistance(org, deCompressed, 5, 1, this.toString());

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixMultChain() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector1 = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 0.9, 1.1, 1.0, 3));

			// ChainType ctype = ChainType.XtwXv;
			// Linear regression .
			for(ChainType ctype : new ChainType[] {ChainType.XtwXv, ChainType.XtXv,
				// ChainType.XtXvy
			}) {

				MatrixBlock vector2 = (ctype == ChainType.XtwXv) ? DataConverter
					.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, 1, 0.9, 1.1, 1.0, 3)) : null;

				// matrix-vector uncompressed
				MatrixBlock ret1 = mb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, _k);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype, _k);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

				if(compressionSettings.lossy) {
					// TODO Make actual calculation to know the tolerance
					// double scaledTolerance = lossyTolerance * d1.length * d1.length * 1.5;
					// if(ctype == ChainType.XtwXv){
					// scaledTolerance *= d1.length * d1.length * 0.5;
					// }
					// TestUtils.compareMatrices(d1, d2, d1.length, d1[0].length, scaledTolerance );
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.95, 0.95, this.toString());
				}
				else {
					if(rows > 50000)
						TestUtils.compareMatricesPercentageDistance(d1, d2, 0.99, 0.999, this.toString());
					else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE ||
						overlappingType == OverLapping.MATRIX_PLUS || overlappingType == OverLapping.MATRIX ||
						overlappingType == OverLapping.COL)
						TestUtils.compareMatricesPercentageDistance(d1, d2, 0.98, 0.99, this.toString());
					else
						TestUtils.compareMatricesBitAvgDistance(d1, d2, 2048, 350, this.toString());

				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testVectorMatrixMult() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock vector = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(1, rows, 0.9, 1.5, 1.0, 3));

		testLeftMatrixMatrix(vector);
	}

	@Test
	public void testLeftMatrixMatrixMultSmall() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock matrix = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(3, rows, 0.9, 1.5, 1.0, 3));

		testLeftMatrixMatrix(matrix);

	}

	@Test
	public void testLeftMatrixMatrixMultMedium() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock matrix = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(50, rows, 0.9, 1.5, 1.0, 3));

		testLeftMatrixMatrix(matrix);
	}

	@Test
	public void testLeftMatrixMatrixMultSparse() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock matrix = DataConverter.convertToMatrixBlock(TestUtils.generateTestMatrix(2, rows, 0.9, 1.5, .1, 3));

		testLeftMatrixMatrix(matrix);
	}

	public void testLeftMatrixMatrix(MatrixBlock matrix) {
		try {
			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// vector-matrix uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(matrix, mb, new MatrixBlock(), abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(matrix, cmb, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			if(compressionSettings.lossy) {
				TestUtils.compareMatricesPercentageDistance(d1, d2, 0.25, 0.83, compressionSettings.toString());
			}
			else {
				if(rows > 65000)
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.99, 0.99, compressionSettings.toString());
				else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE ||
					overlappingType == OverLapping.MATRIX_PLUS || overlappingType == OverLapping.MATRIX ||
					overlappingType == OverLapping.COL)
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 1500000, 1000, this.toString());
				else
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 24000, 512, compressionSettings.toString());

			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixVectorMult01() {
		testMatrixVectorMult(1.0, 1.1);
	}

	@Test
	public void testMatrixVectorMult02() {
		testMatrixVectorMult(0.7, 1.0);
	}

	@Test
	public void testMatrixVectorMult03() {
		testMatrixVectorMult(-1.0, 1.0);
	}

	@Test
	public void testMatrixVectorMult04() {
		testMatrixVectorMult(1.0, 5.0);
	}

	public void testMatrixVectorMult(double min, double max) {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock vector = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, min, max, 1.0, 3));
		testRightMatrixMatrix(vector);
	}

	@Test
	public void testRightMatrixMatrixMultSmall() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock matrix = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 2, 0.9, 1.5, 1.0, 3));

		testRightMatrixMatrix(matrix);
	}

	@Test
	public void testRightMatrixMatrixMultMedium() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock matrix = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 16, 0.9, 1.5, 1.0, 3));
		testRightMatrixMatrix(matrix);

	}

	@Test
	public void testRightMatrixMatrixMultSparse() {

		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test

		MatrixBlock matrix = DataConverter
			.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 25, 0.9, 1.5, 0.2, 3));
		testRightMatrixMatrix(matrix);
	}

	public void testRightMatrixMatrix(MatrixBlock matrix) {
		try {
			matrix.quickSetValue(0, 0, 10);
			// Make Operator
			AggregateBinaryOperator abop = InstructionUtils.getMatMultOperator(_k);

			// vector-matrix uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(mb, matrix, new MatrixBlock(), abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(cmb, matrix, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			if(compressionSettings.lossy) {
				if(rows > 65000) {
					TestUtils.compareMatrices(d1, d2, lossyTolerance * cols * rows / 10, this.toString());
				}
				else {
					TestUtils.compareMatrices(d1, d2, lossyTolerance * cols * rows / 15, this.toString());

				}
			}
			else {
				if(rows > 65000)
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.5, 0.99, compressionSettings.toString());
				else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE ||
					overlappingType == OverLapping.MATRIX_PLUS || overlappingType == OverLapping.MATRIX ||
					overlappingType == OverLapping.COL)
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 1600000, 1000, this.toString());
				else
					TestUtils.compareMatricesBitAvgDistance(d1, d2, 10000, 500, compressionSettings.toString());

			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testTransposeSelfMatrixMult() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// ChainType ctype = ChainType.XtwXv;
			for(MMTSJType mType : new MMTSJType[] {MMTSJType.LEFT,
				// MMTSJType.RIGHT
			}) {
				// matrix-vector uncompressed
				MatrixBlock ret1 = mb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, _k);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType, _k);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				// High probability that The value is off by some amount
				if(compressionSettings.lossy) {
					// Probably the worst thing you can do to increase the amount the values are estimated wrong
					TestUtils.compareMatricesPercentageDistance(d1, d2, 0.5, 0.8, this.toString());
				}
				else {
					if(rows > 50000) {
						TestUtils
							.compareMatricesPercentageDistance(d1, d2, 0.99, 0.999, this.toString());
					}
					else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE ||
						overlappingType == OverLapping.MATRIX_PLUS || overlappingType == OverLapping.MATRIX ||
						overlappingType == OverLapping.COL)
						TestUtils.compareMatricesPercentageDistance(d1, d2, 0.98, 0.98, this.toString());
					else {
						TestUtils.compareMatricesBitAvgDistance(d1, d2, 2048, 512, this.toString());
					}
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testScalarOpRightMultiplyPositive() {
		double mult = 7;
		ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
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
	public void testScalarRightOpSubtract() {
		double addValue = 15;
		ScalarOperator sop = new RightScalarOperator(Minus.getMinusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpLess() {
		double addValue = 0.11;
		ScalarOperator sop = new RightScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpGreater() {
		double addValue = 0.11;
		ScalarOperator sop = new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarRightOpPower2() {
		double addValue = 2;
		ScalarOperator sop = new RightScalarOperator(Power2.getPower2FnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarOpLeftMultiplyPositive() {
		double mult = 7;
		ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarOpLeftMultiplyNegative() {
		double mult = -7;
		ScalarOperator sop = new LeftScalarOperator(Multiply.getMultiplyFnObject(), mult, _k);
		testScalarOperations(sop, lossyTolerance * 7);
	}

	@Test
	public void testScalarLeftOpAddition() {
		double addValue = 4;
		ScalarOperator sop = new LeftScalarOperator(Plus.getPlusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.05);
	}

	@Test
	public void testScalarLeftOpSubtract() {
		double addValue = 15;
		ScalarOperator sop = new LeftScalarOperator(Minus.getMinusFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpLess() {
		double addValue = 0.11;
		ScalarOperator sop = new LeftScalarOperator(LessThanEquals.getLessThanEqualsFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	@Test
	public void testScalarLeftOpGreater() {
		double addValue = 0.11;
		ScalarOperator sop = new LeftScalarOperator(GreaterThan.getGreaterThanFnObject(), addValue);
		testScalarOperations(sop, lossyTolerance + 0.1);
	}

	// @Test
	// This test does not make sense to execute... since the result of left power always is 4.
	// Furthermore it does not work consistently in our normal matrix blocks ... and should never be used.
	// public void testScalarLeftOpPower2() {
	// double addValue = 2;
	// ScalarOperator sop = new LeftScalarOperator(Power2.getPower2FnObject(), addValue);
	// testScalarOperations(sop, lossyTolerance + 0.1);
	// }

	public void testScalarOperations(ScalarOperator sop, double tolerance) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock ret1 = mb.scalarOperations(sop, new MatrixBlock());

			// matrix-scalar compressed
			MatrixBlock ret2 = cmb.scalarOperations(sop, new MatrixBlock());

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

			if(compressionSettings.lossy)
				TestUtils.compareMatrices(d1, d2, tolerance, this.toString());
			else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE || overlappingType == OverLapping.MATRIX_PLUS ||
				overlappingType == OverLapping.MATRIX || overlappingType == OverLapping.COL)
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 50000, 1000, this.toString());
			else
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 150, 1, this.toString());

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testBinaryMVAddition() {

		ValueFunction vf = Plus.getPlusFnObject();
		testBinaryMV(vf);
	}

	@Test
	public void testBinaryMVMultiply() {
		ValueFunction vf = Multiply.getMultiplyFnObject();
		testBinaryMV(vf);
	}

	@Test
	public void testBinaryMVMinus() {
		ValueFunction vf = Minus.getMinusFnObject();
		testBinaryMV(vf);
	}

	@Test
	public void testBinaryMVXor() {
		ValueFunction vf = Xor.getXorFnObject();
		testBinaryMV(vf);
	}

	public void testBinaryMV(ValueFunction vf) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			BinaryOperator bop = new BinaryOperator(vf);
			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(1, cols, -1.0, 1.5, 1.0, 3));

			MatrixBlock ret1 = mb.binaryOperations(bop, vector, new MatrixBlock());
			MatrixBlock ret2 = cmb.binaryOperations(bop, vector, new MatrixBlock());
			if(ret2 instanceof CompressedMatrixBlock)
				ret2 = ((CompressedMatrixBlock) ret2).decompress();
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

			if(compressionSettings.lossy)
				TestUtils.compareMatrices(d1, d2, lossyTolerance * 2, this.toString());
			else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE || overlappingType == OverLapping.MATRIX_PLUS ||
				overlappingType == OverLapping.MATRIX || overlappingType == OverLapping.COL)
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 65536, 512, this.toString());
			else
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 150, 1, this.toString());

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testBinaryVMMultiply() {
		ValueFunction vf = Multiply.getMultiplyFnObject();
		testBinaryVM(vf);
	}

	public void testBinaryVM(ValueFunction vf) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			// NOTE THIS METHOD DECOMPRESSES AND MULTIPLIES

			BinaryOperator bop = new BinaryOperator(vf);
			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, 1, -1.0, 1.5, 1.0, 3));

			MatrixBlock ret1 = mb.binaryOperations(bop, vector, new MatrixBlock());
			MatrixBlock ret2 = cmb.binaryOperations(bop, vector, new MatrixBlock());

			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

			if(compressionSettings.lossy)
				TestUtils.compareMatrices(d1, d2, lossyTolerance * 2, this.toString());
			else if(overlappingType == OverLapping.MATRIX_MULT_NEGATIVE || overlappingType == OverLapping.MATRIX_PLUS ||
				overlappingType == OverLapping.MATRIX || overlappingType == OverLapping.COL)
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 65536, 512, this.toString());
			else
				TestUtils.compareMatricesBitAvgDistance(d1, d2, 150, 1, this.toString());

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}
}
