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

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;

import org.apache.sysds.lops.MMTSJ.MMTSJType;
import org.apache.sysds.lops.MapMultChain.ChainType;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
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
		CompressionSettings compSettings, MatrixTypology matrixTypology) {
		super(sparType, valType, valRange, compSettings, matrixTypology);
	}

	@Test
	public void testConstruction() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock)) {
				return; // Input was not compressed then just pass test
				// Assert.assertTrue("Compression Failed \n" + this.toString(), false);
			}

			TestUtils.compareMatricesBitAvgDistance(input, deCompressed, rows, cols, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testGetValue() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			for(int i = 0; i < rows; i++)
				for(int j = 0; j < cols; j++) {
					double ulaVal = input[i][j];
					double claVal = cmb.getValue(i, j); // calls quickGetValue internally
					TestUtils.compareScalarBitsJUnit(ulaVal, claVal, 0); // Should be exactly same value
				}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testAppend() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, 1, 1, 1, 1.0, 3));

			// matrix-vector uncompressed
			MatrixBlock ret1 = mb.append(vector, new MatrixBlock());

			// matrix-vector compressed
			MatrixBlock ret2 = cmb.append(vector, new MatrixBlock());
			if(ret2 instanceof CompressedMatrixBlock)
				ret2 = ((CompressedMatrixBlock) ret2).decompress();

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatricesBitAvgDistance(d1, d2, rows, cols + 1, 0, 1);
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
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 0, 1, 1.0, 3));

			// ChainType ctype = ChainType.XtwXv;
			// Linear regression .
			for(ChainType ctype : new ChainType[] {ChainType.XtwXv, ChainType.XtXv,
				// ChainType.XtXvy
			}) {

				MatrixBlock vector2 = (ctype == ChainType.XtwXv) ? DataConverter
					.convertToMatrixBlock(TestUtils.generateTestMatrix(rows, 1, 0, 1, 1.0, 3)) : null;

				// matrix-vector uncompressed
				MatrixBlock ret1 = mb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.chainMatrixMultOperations(vector1, vector2, new MatrixBlock(), ctype);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				TestUtils.compareMatricesBitAvgDistance(d1, d2, cols, 1, 512, 32);
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
				MatrixBlock ret1 = mb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType);

				// matrix-vector compressed
				MatrixBlock ret2 = cmb.transposeSelfMatrixMultOperations(new MatrixBlock(), mType);

				// compare result with input
				double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
				double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
				TestUtils.compareMatricesBitAvgDistance(d1, d2, cols, cols, 2048, 20);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testMatrixVectorMult() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(cols, 1, 1, 1, 1.0, 3));

			// Make Operator
			AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator abop = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), aop);

			// matrix-vector uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(mb, vector, new MatrixBlock(), abop);

			// matrix-vector compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(cmb, vector, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatricesBitAvgDistance(d1, d2, rows, 1, 1024, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testVectorMatrixMult() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			MatrixBlock vector = DataConverter
				.convertToMatrixBlock(TestUtils.generateTestMatrix(1, rows, 1, 1, 1.0, 3));

			// Make Operator
			AggregateOperator aop = new AggregateOperator(0, Plus.getPlusFnObject());
			AggregateBinaryOperator abop = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), aop);

			// vector-matrix uncompressed
			MatrixBlock ret1 = mb.aggregateBinaryOperations(vector, mb, new MatrixBlock(), abop);

			// vector-matrix compressed
			MatrixBlock ret2 = cmb.aggregateBinaryOperations(vector, cmb, new MatrixBlock(), abop);

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);
			TestUtils.compareMatricesBitAvgDistance(d1, d2, 1, cols, 10000, 500);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testScalarOperationsSparseUnsafe() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			// matrix-scalar uncompressed
			ScalarOperator sop = new RightScalarOperator(Plus.getPlusFnObject(), 7);
			MatrixBlock ret1 = mb.scalarOperations(sop, new MatrixBlock());

			// matrix-scalar compressed
			MatrixBlock ret2 = cmb.scalarOperations(sop, new MatrixBlock());
			if(ret2 instanceof CompressedMatrixBlock)
				ret2 = ((CompressedMatrixBlock) ret2).decompress();

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

			TestUtils.compareMatricesBitAvgDistance(d1, d2, rows, cols, 150, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testScalarOperations() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return; // Input was not compressed then just pass test

			// matrix-scalar uncompressed
			ScalarOperator sop = new RightScalarOperator(Multiply.getMultiplyFnObject(), 7);
			MatrixBlock ret1 = mb.scalarOperations(sop, new MatrixBlock());

			// matrix-scalar compressed
			MatrixBlock ret2 = cmb.scalarOperations(sop, new MatrixBlock());
			if(ret2 instanceof CompressedMatrixBlock)
				ret2 = ((CompressedMatrixBlock) ret2).decompress();

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(ret1);
			double[][] d2 = DataConverter.convertToDoubleMatrix(ret2);

			TestUtils.compareMatricesBitAvgDistance(d1, d2, rows, cols, 150, 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Override
	public void testUnaryOperators(AggType aggType) {
		AggregateUnaryOperator auop = super.getUnaryOperator(aggType, 1);
		testUnaryOperators(aggType, auop);
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
			CompressedMatrixBlock cmb2 = new CompressedMatrixBlock();
			cmb2.readFields(fis);

			// decompress the compressed matrix block
			MatrixBlock tmp = cmb2.decompress();

			// compare result with input
			double[][] d1 = DataConverter.convertToDoubleMatrix(mb);
			double[][] d2 = DataConverter.convertToDoubleMatrix(tmp);

			TestUtils.compareMatricesBitAvgDistance(d1, d2, rows, cols, 0, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionRatio() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			CompressionStatistics cStat = ((CompressedMatrixBlock) cmb).getCompressionStatistics();
			assertTrue("Compression ration if compressed should be larger than 1", cStat.ratio > 1);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionEstimationVSCompression() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			CompressionStatistics cStat = ((CompressedMatrixBlock) cmb).getCompressionStatistics();
			long colsEstimate = cStat.estimatedSizeCols;
			long actualSize = cStat.size;
			long originalSize = cStat.originalSize;
			int allowedTolerance = 0;

			if(compressionSettings.samplingRatio < 1.0) {
				allowedTolerance = sampleTolerance;
			}

			StringBuilder builder = new StringBuilder();
			builder.append("\n\t" + String.format("%-40s - %12d", "Actual compressed size: ", actualSize));
			builder.append("\n\t" + String.format("%-40s - %12d with tolerance: %5d",
				"<= estimated isolated ColGroups: ",
				colsEstimate,
				allowedTolerance));
			builder.append("\n\t" + String.format("%-40s - %12d", "<= Original size: ", originalSize));
			builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
			builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
			builder.append("\n\t" + this.toString());
			boolean res = actualSize - colsEstimate <= allowedTolerance;
			assertTrue(builder.toString(), res);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionEstimationVSJolEstimate() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			CompressionStatistics cStat = ((CompressedMatrixBlock) cmb).getCompressionStatistics();
			long actualSize = cStat.size;
			long originalSize = cStat.originalSize;
			long JolEstimatedSize = getJolSize(((CompressedMatrixBlock) cmb));

			StringBuilder builder = new StringBuilder();
			builder.append("\n\t" + String.format("%-40s - %12d", "Actual compressed size: ", actualSize));
			builder.append("\n\t" + String.format("%-40s - %12d", "<= Original size: ", originalSize));
			builder.append("\n\t" + String.format("%-40s - %12d", "and equal to JOL Size: ", JolEstimatedSize));
			// builder.append("\n\t " + getJolSizeString(cmb));
			builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
			builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
			builder.append("\n\t" + this.toString());

			//NOTE: The Jol estimate is wrong for shared dictionaries because
			//      it treats the object hierarchy as a tree and not a graph
			assertTrue(builder.toString(), actualSize <= originalSize 
				&& (compressionSettings.allowSharedDDCDictionary || actualSize == JolEstimatedSize));
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	@Test
	public void testCompressionScale() {
		// This test is here for a sanity check such that we verify that the compression ratio from our Matrix
		// Compressed Block is not unreasonably good.
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;

			CompressionStatistics cStat = ((CompressedMatrixBlock) cmb).getCompressionStatistics();

			double compressRatio = cStat.ratio;
			long actualSize = cStat.size;
			long originalSize = cStat.originalSize;

			StringBuilder builder = new StringBuilder();
			builder.append("Compression Ratio sounds suspiciously good at: " + compressRatio);
			builder.append("\n\tActual compressed size: " + actualSize);
			builder.append(" original size: " + originalSize);
			builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
			builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
			builder.append("\n\t" + this.toString());

			assertTrue(builder.toString(), compressRatio < 1000.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	private static long getJolSize(CompressedMatrixBlock cmb) {
		Layouter l = new HotSpotLayouter(new X86_64_DataModel());
		long jolEstimate = 0;
		CompressionStatistics cStat = cmb.getCompressionStatistics();
		for(Object ob : new Object[] {cmb, cStat, cStat.getColGroups(), cStat.getTimeArrayList(), cmb.getColGroups()}) {
			jolEstimate += ClassLayout.parseInstance(ob, l).instanceSize();
		}
		for(ColGroup cg : cmb.getColGroups()) {
			jolEstimate += cg.estimateInMemorySize();
		}
		return jolEstimate;
	}

	@SuppressWarnings("unused")
	private static String getJolSizeString(CompressedMatrixBlock cmb) {
		StringBuilder builder = new StringBuilder();
		Layouter l = new HotSpotLayouter(new X86_64_DataModel());
		long diff;
		long jolEstimate = 0;
		CompressionStatistics cStat = cmb.getCompressionStatistics();
		for(Object ob : new Object[] {cmb, cStat, cStat.getColGroups(), cStat.getTimeArrayList(), cmb.getColGroups()}) {
			ClassLayout cl = ClassLayout.parseInstance(ob, l);
			diff = cl.instanceSize();
			jolEstimate += diff;
			builder.append(cl.toPrintable());
			builder.append("TOTAL MEM: " + jolEstimate + " diff " + diff + "\n");
		}
		for(ColGroup cg : cmb.getColGroups()) {
			diff = cg.estimateInMemorySize();
			jolEstimate += diff;
			builder.append(cg.getCompType());
			builder.append("TOTAL MEM: " + jolEstimate + " diff " + diff + "\n");
		}
		return builder.toString();
	}
}
