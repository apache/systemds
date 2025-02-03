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

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Collection;
import java.util.Random;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysds.common.Types.CorrectionLocationType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.cost.CostEstimatorBuilder;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Minus1Multiply;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.PlusMultiply;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateTernaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.OverLapping;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.openjdk.jol.datamodel.X86_64_DataModel;
import org.openjdk.jol.info.ClassLayout;
import org.openjdk.jol.layouters.HotSpotLayouter;
import org.openjdk.jol.layouters.Layouter;

@RunWith(value = Parameterized.class)
public class CompressedMatrixTest extends AbstractCompressedUnaryTests {

	public CompressedMatrixTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettingsBuilder compSettings, MatrixTypology matrixTypology, OverLapping ov,
		Collection<CompressionType> ct, CostEstimatorBuilder csb) {
		super(sparType, valType, valRange, compSettings, matrixTypology, ov, 1, ct, csb);
	}

	@Test
	public void testGetValue() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			Random r = new Random();
			final int min = r.nextInt(rows);
			final int max = Math.min(r.nextInt(rows - min) + min, min + 1000);
			for(int i = min; i < max; i++)
				for(int j = 0; j < cols; j++) {
					final double ulaVal = mb.get(i, j);
					final double claVal = cmb.get(i, j); // calls quickGetValue internally
					if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
						assertTrue(bufferedToString, TestUtils.compareCellValue(ulaVal, claVal, lossyTolerance, false));
					else if(OverLapping.effectOnOutput(overlappingType))
						assertTrue(bufferedToString, TestUtils.getPercentDistance(ulaVal, claVal, true) > .99);
					else
						TestUtils.compareScalarBitsJUnit(ulaVal, claVal, 0, bufferedToString);
				}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testQuickGetValue() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test
			Random r = new Random();
			final int min = r.nextInt(rows);
			final int max = Math.min(r.nextInt(rows - min) + min, min + 1000);

			for(int i = min; i < max; i++)
				for(int j = 0; j < cols; j++) {
					final double ulaVal = mb.get(i, j);
					final double claVal = cmb.get(i, j); // calls quickGetValue internally
					if(_cs != null && (_cs.lossy || overlappingType == OverLapping.SQUASH))
						assertTrue(bufferedToString, TestUtils.compareCellValue(ulaVal, claVal, lossyTolerance, false));
					else if(OverLapping.effectOnOutput(overlappingType))
						assertTrue(bufferedToString, TestUtils.getPercentDistance(ulaVal, claVal, true) > .99);
					else
						TestUtils.compareScalarBitsJUnit(ulaVal, claVal, 0, bufferedToString); // Should be exactly same value
				}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Override
	public void testUnaryOperators(AggType aggType, boolean inCP) {
		AggregateUnaryOperator auop = super.getUnaryOperator(aggType, 1);
		testUnaryOperators(aggType, auop, inCP);
	}

	@Test
	public void testNonZeros() {
		if(!(cmb instanceof CompressedMatrixBlock))
			return; // Input was not compressed then just pass test
		if(!(cmb.getNonZeros() >= mb.getNonZeros() || cmb.getNonZeros() == -1)) { // guarantee that the nnz is at least
																											// the nnz
			fail(bufferedToString + "\nIncorrect number of non Zeros should guarantee greater than or equals but are "
				+ cmb.getNonZeros() + " and should be: " + mb.getNonZeros());
		}
	}

	@Test
	public void testSerialization() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			// serialize compressed matrix block
			ByteArrayOutputStream bos = new ByteArrayOutputStream();
			DataOutputStream fos = new DataOutputStream(bos);
			cmb.write(fos);
			// deserialize compressed matrix block
			ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
			DataInputStream fis = new DataInputStream(bis);
			CompressedMatrixBlock cmb2 = new CompressedMatrixBlock(-1, -1);
			cmb2.readFields(fis);

			// decompress the compressed matrix block
			MatrixBlock tmp = cmb2.decompress();

			((CompressedMatrixBlock) cmb).clearSoftReferenceToDecompressed();

			compareResultMatrices(mb, tmp, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionRatio() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || _cs == null)
				return;
			CompressionStatistics cStat = cmbStats;
			if(cStat != null && !(mb.isEmpty()) && cStat.originalSize > 1000)
				assertTrue("Compression ration if compressed should be larger than 1", cStat.getRatio() > 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionEstimationVSCompression() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || _cs == null)
				return;
			CompressionStatistics cStat = cmbStats;
			if(cStat != null && !(mb.isEmpty()) && cStat.originalSize > 1000) {
				long colsEstimate = cStat.estimatedSizeCols;
				long actualSize = cStat.compressedSize;
				long originalSize = cStat.originalSize;
				int allowedTolerance = 4096;

				if(_cs.samplingRatio < 1.0) {
					allowedTolerance = sampleTolerance;
				}
				if(rows > 50000) {
					allowedTolerance *= 10;
				}

				boolean res = Math.abs(colsEstimate - actualSize) <= originalSize;
				res = res && actualSize - allowedTolerance < colsEstimate;
				if(!res) {
					StringBuilder builder = new StringBuilder();
					builder.append("\n\t" + String.format("%-40s - %12d", "Actual compressed size: ", actualSize));
					builder.append("\n\t" + String.format("%-40s - %12d with tolerance: %5d",
						"<= estimated isolated ColGroups: ", colsEstimate, allowedTolerance));
					builder.append("\n\t" + String.format("%-40s - %12d", "<= Original size: ", originalSize));
					builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
					builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
					builder.append("\n\t" + bufferedToString);
					fail(builder.toString());
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Ignore
	@Test
	public void testCompressionEstimationVSJolEstimate() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			CompressionStatistics cStat = cmbStats;
			if(cStat != null) {
				long actualSize = cStat.compressedSize;
				long originalSize = cStat.originalSize;
				long JolEstimatedSize = getJolSize(((CompressedMatrixBlock) cmb), cmbStats);

				StringBuilder builder = new StringBuilder();
				if(!(actualSize <= originalSize && (_cs.allowSharedDictionary || actualSize == JolEstimatedSize))) {

					builder.append("\n\t" + String.format("%-40s - %12d", "Actual compressed size: ", actualSize));
					builder.append("\n\t" + String.format("%-40s - %12d", "<= Original size: ", originalSize));
					builder.append("\n\t" + String.format("%-40s - %12d", "and equal to JOL Size: ", JolEstimatedSize));
					// builder.append("\n\t " + getJolSizeString(cmb));
					builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
					builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
					builder.append("\n\t" + bufferedToString);

					// NOTE: The Jol estimate is wrong for shared dictionaries because
					// it treats the object hierarchy as a tree and not a graph
					fail(builder.toString());
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionScale() {
		// This test is here for a sanity check such that we verify that the compression
		// ratio from our Matrix
		// Compressed Block is not unreasonably good.
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;

			CompressionStatistics cStat = cmbStats;
			if(cStat != null) {
				final double compressRatio = cStat.getRatio();

				if(compressRatio > 1000.0) {
					StringBuilder builder = new StringBuilder();
					builder.append("Compression Ratio sounds suspiciously good at: " + compressRatio);
					builder.append("\n\tActual compressed size: " + cStat.compressedSize);
					builder.append(" original size: " + cStat.originalSize);
					builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
					builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
					builder.append("\n\t" + bufferedToString);
					fail(builder.toString());
				}
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(bufferedToString + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testContainsValue_maybe() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			assertTrue(bufferedToString, cmb.containsValue(min) == mb.containsValue(min));
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testContainsValue_not() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			boolean ret1 = cmb.containsValue(min - 1);
			boolean ret2 = mb.containsValue(min - 1);
			assertTrue(bufferedToString, ret1 == ret2);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testCompressedMatrixConstruction() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			CompressedMatrixBlock cmbC = (CompressedMatrixBlock) cmb;
			CompressedMatrixBlock cmbCopy = new CompressedMatrixBlock(cmbC);

			compareResultMatrices(mb, cmbCopy, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testCompressedMatrixCopy() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			CompressedMatrixBlock cmbCopy = new CompressedMatrixBlock(cmb.getNumRows(), cmb.getNumColumns());
			cmbCopy.copy(cmb);

			compareResultMatrices(mb, cmbCopy, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void testCompressedMatrixCopyMatrixBlock_shouldThrowException() {
		CompressedMatrixBlock cmbCopy = new CompressedMatrixBlock(mb.getNumRows(), mb.getNumColumns());
		cmbCopy.copy(mb);
	}

	@Test(expected = RuntimeException.class)
	public void testCompressedMatrixCopyToSelf_shouldThrowException() {
		cmb.copy(cmb);
	}

	@Test
	public void testAggregateTernaryOperation() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 1000)
				return;
			CorrectionLocationType corr = CorrectionLocationType.LASTCOLUMN;
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corr);
			AggregateTernaryOperator op = new AggregateTernaryOperator(Multiply.getMultiplyFnObject(), agg,
				ReduceAll.getReduceAllFnObject());

			int nrow = mb.getNumRows();
			int ncol = mb.getNumColumns();

			MatrixBlock m2 = new MatrixBlock(nrow, ncol, 13.0);
			MatrixBlock m3 = new MatrixBlock(nrow, ncol, 14.0);

			MatrixBlock ret1 = MatrixBlock.aggregateTernaryOperations(cmb, m2, m3, null, op, true);
			ucRet = MatrixBlock.aggregateTernaryOperations(mb, m2, m3, ucRet, op, true);

			compareResultMatrices(ucRet, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testAggregateTernaryOperationZero() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			CorrectionLocationType corr = CorrectionLocationType.LASTCOLUMN;
			AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), corr);
			AggregateTernaryOperator op = new AggregateTernaryOperator(Multiply.getMultiplyFnObject(), agg,
				ReduceAll.getReduceAllFnObject());

			int nrow = mb.getNumRows();
			int ncol = mb.getNumColumns();

			MatrixBlock m2 = new MatrixBlock(nrow, ncol, 0);
			MatrixBlock m3 = new MatrixBlock(nrow, ncol, 14.0);

			MatrixBlock ret1 = MatrixBlock.aggregateTernaryOperations(cmb, m2, m3, null, op, true);
			ucRet = MatrixBlock.aggregateTernaryOperations(mb, m2, m3, ucRet, op, true);

			compareResultMatrices(ucRet, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testTernaryOperation() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock) || rows * cols > 10000)
				return;
			TernaryOperator op = new TernaryOperator(PlusMultiply.getFnObject(), _k);

			int nrow = mb.getNumRows();
			int ncol = mb.getNumColumns();

			MatrixBlock m2 = new MatrixBlock(1, 1, 0);
			MatrixBlock m3 = new MatrixBlock(nrow, ncol, 14.0);
			MatrixBlock ret1 = cmb.ternaryOperations(op, m2, m3, new MatrixBlock());
			ucRet = mb.ternaryOperations(op, m2, m3, ucRet);

			compareResultMatrices(ucRet, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testBinaryEmptyScalarOp() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject());
			op.setNumThreads(_k);

			MatrixBlock m2 = new MatrixBlock(1, 1, 0);
			MatrixBlock ret1 = cmb.binaryOperations(op, m2, new MatrixBlock());
			ScalarOperator sop = new RightScalarOperator(op.fn, m2.get(0, 0), op.getNumThreads());
			ucRet = mb.scalarOperations(sop, ucRet);

			compareResultMatrices(ucRet, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testBinaryEmptyMatrixMultiplicationOp() {
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject(), _k);
		testBinaryEmptyMatrixOp(op);
	}

	@Test
	public void testBinaryEmptyMatrixMinusOp() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), _k);
		testBinaryEmptyMatrixOp(op);
	}

	@Test
	public void testBinaryEmptyMatrixPlusOp() {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), _k);
		testBinaryEmptyMatrixOp(op);
	}

	@Test
	public void testBinaryEmptyMatrixMinusMultiplyOp() {
		BinaryOperator op = new BinaryOperator(new MinusMultiply(42), _k);
		op.setNumThreads(_k);
		testBinaryEmptyMatrixOp(op);
	}

	@Test
	public void testBinaryEmptyMatrixMinus1MultiplyOp() {
		BinaryOperator op = new BinaryOperator(Minus1Multiply.getMinus1MultiplyFnObject(), _k);
		testBinaryEmptyMatrixOp(op);
	}

	public void testBinaryEmptyMatrixOp(BinaryOperator op) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;

			MatrixBlock m2 = new MatrixBlock(cmb.getNumRows(), cmb.getNumColumns(), 0);
			MatrixBlock ret1 = cmb.binaryOperations(op, m2, new MatrixBlock());
			ucRet = mb.binaryOperations(op, m2, ucRet);

			compareResultMatrices(ucRet, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException(e);
		}
	}

	@Test
	public void testBinaryEmptyRowVectorMultiplicationOp() {
		BinaryOperator op = new BinaryOperator(Multiply.getMultiplyFnObject(), _k);
		testBinaryEmptyRowVectorOp(op);
	}

	@Test
	public void testBinaryEmptyRowVectorMinusOp() {
		BinaryOperator op = new BinaryOperator(Minus.getMinusFnObject(), _k);
		testBinaryEmptyRowVectorOp(op);
	}

	@Test
	public void testBinaryEmptyRowVectorPlusOp() {
		BinaryOperator op = new BinaryOperator(Plus.getPlusFnObject(), _k);
		testBinaryEmptyRowVectorOp(op);
	}

	public void testBinaryEmptyRowVectorOp(BinaryOperator op) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;

			MatrixBlock m2 = new MatrixBlock(1, cmb.getNumColumns(), 0);
			MatrixBlock ret1 = cmb.binaryOperations(op, m2, new MatrixBlock());
			ucRet = mb.binaryOperations(op, m2, ucRet);
			compareResultMatrices(ucRet, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testDecompressToNoOffsets() {
		testDecompressToMatrixWithOffsets(0, 0, rows + 1, cols + 1);
	}

	@Test
	public void testDecompressToOffset1Row() {
		testDecompressToMatrixWithOffsets(1, 0, rows + 1, cols + 1);
	}

	@Test
	public void testDecompressToOffset1Col() {
		testDecompressToMatrixWithOffsets(0, 1, rows + 1, cols + 1);
	}

	@Test
	public void testDecompressToOffsetBoth1() {
		testDecompressToMatrixWithOffsets(1, 1, rows + 1, cols + 1);
	}

	@Test
	public void testDecompressToMiddleOfMatrix() {
		testDecompressToMatrixWithOffsets(10, 10, rows + 20, cols + 20);
	}

	public void testDecompressToMatrixWithOffsets(int rowOff, int colOff, int outRows, int outCols) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;

			MatrixBlock ret1 = new MatrixBlock(outRows, outCols, false);
			ret1.allocateDenseBlock();
			MatrixBlock ret2 = new MatrixBlock(outRows, outCols, false);
			ret2.allocateDenseBlock();

			cmb.putInto(ret1, rowOff, colOff, false);
			ret1.setNonZeros(cmb.getNonZeros());

			mb.putInto(ret2, rowOff, colOff, false);
			ret2.setNonZeros(mb.getNonZeros());

			compareResultMatrices(ret2, ret1, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	@Test
	public void testDecompressToOffsetSparseExtraColumns() {
		if(cmb instanceof CompressedMatrixBlock && !((CompressedMatrixBlock) cmb).isOverlapping())
			testsDecompressToSparseMatrixWithOffsets(10, 10, rows + 100, cols + 1000);
	}

	@Test
	public void testDecompressToOffsetSparseExtraRows() {
		if(cmb instanceof CompressedMatrixBlock && !((CompressedMatrixBlock) cmb).isOverlapping())
			testsDecompressToSparseMatrixWithOffsets(10, 10, rows + 10000, cols + 10);
	}

	public void testsDecompressToSparseMatrixWithOffsets(int rowOff, int colOff, int outRows, int outCols) {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;

			MatrixBlock ret1 = new MatrixBlock(outRows, outCols, true);
			ret1.allocateSparseRowsBlock();
			MatrixBlock ret2 = new MatrixBlock(outRows, outCols, true);
			ret2.allocateSparseRowsBlock();

			cmb.putInto(ret1, rowOff, colOff, false);
			ret1.setNonZeros(cmb.getNonZeros());
			if(ret1.isInSparseFormat())
				ret1.sortSparseRows();

			mb.putInto(ret2, rowOff, colOff, false);
			ret2.setNonZeros(mb.getNonZeros());
			if(ret2.isInSparseFormat())
				ret2.sortSparseRows();

			compareResultMatrices(ret2, ret1, 1);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw e;
		}
	}

	private static long getJolSize(CompressedMatrixBlock cmb, CompressionStatistics cStat) {
		Layouter l = new HotSpotLayouter(new X86_64_DataModel());
		long jolEstimate = 0;
		for(Object ob : new Object[] {cmb, cmb.getColGroups()}) {
			jolEstimate += ClassLayout.parseInstance(ob, l).instanceSize();
		}
		for(AColGroup cg : cmb.getColGroups()) {
			jolEstimate += cg.estimateInMemorySize();
		}
		return jolEstimate;
	}

	@Test
	public void toRDDAndBack100() {
		toRDDAndBack(100);
	}

	@Test
	public void toRDDAndBack200() {
		toRDDAndBack(200);
	}

	@Test
	public void toRDDAndBack500() {
		toRDDAndBack(500);
	}

	@Test
	public void toRDDAndBack1000() {
		toRDDAndBack(1000);
	}

	public void toRDDAndBack(int blen) {
		try {
			JavaSparkContext sc = SparkExecutionContext.getSparkContextStatic();
			JavaPairRDD<MatrixIndexes, MatrixBlock> rdd = SparkExecutionContext.toMatrixJavaPairRDD(sc, cmb, blen);
			MatrixBlock back = SparkExecutionContext.toMatrixBlock(rdd, mb.getNumRows(), mb.getNumColumns(), blen,
				mb.getNonZeros());
			compareResultMatrices(back, mb, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
