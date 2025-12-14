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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.EnumSet;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDeltaDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupIO;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.ComEstExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

public class ColGroupDeltaDDCTest {

	protected static final Log LOG = LogFactory.getLog(ColGroupDeltaDDCTest.class.getName());

	@Test
	public void testDecompressToDenseBlockSingleColumn() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}}, false);
	}

	@Test(expected = NotImplementedException.class)
	public void testDecompressToDenseBlockSingleColumnTransposed() {
		testDecompressToDenseBlock(new double[][] {{1}, {2}, {3}, {4}, {5}}, true);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumns() {
		testDecompressToDenseBlock(new double[][] {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}, false);
	}

	@Test(expected = NotImplementedException.class)
	public void testDecompressToDenseBlockTwoColumnsTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}, {1, 1, 1, 1, 1}}, true);
	}

	@Test
	public void testDecompressToDenseBlockPartialRangeSingleColumn() {
		testDecompressToDenseBlockPartialRange(new double[][] {{1}, {2}, {3}, {4}, {5}}, false, 2, 5);
	}

	@Test
	public void testDecompressToDenseBlockPartialRangeTwoColumns() {
		testDecompressToDenseBlockPartialRange(new double[][] {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}, false, 1, 4);
	}

	@Test
	public void testDecompressToDenseBlockPartialRangeFromMiddle() {
		testDecompressToDenseBlockPartialRange(new double[][] {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}, false, 3, 6);
	}

	public void testDecompressToDenseBlock(double[][] data, boolean isTransposed) {
		if(isTransposed) {
			throw new NotImplementedException("Delta encoding for transposed matrices not yet implemented");
		}
		
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		try {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC))
				.setPreferDeltaEncoding(true)
				.setTransposeInput("false");
			CompressionSettings cs = csb.create();

			final CompressedSizeInfoColGroup cgi = new ComEstExact(mbt, cs).getColGroupInfo(colIndexes);
			CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

			MatrixBlock ret = new MatrixBlock(numRows, numCols, false);
			ret.allocateDenseBlock();
			cg.decompressToDenseBlock(ret.getDenseBlock(), 0, numRows);

			MatrixBlock expected = DataConverter.convertToMatrixBlock(data);
			assertArrayEquals(expected.getDenseBlockValues(), ret.getDenseBlockValues(), 0.01);

		}
		catch(NotImplementedException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName(), e);
		}
	}

	public void testDecompressToDenseBlockPartialRange(double[][] data, boolean isTransposed, int rl, int ru) {
		if(isTransposed) {
			throw new NotImplementedException("Delta encoding for transposed matrices not yet implemented");
		}
		
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		try {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC))
				.setPreferDeltaEncoding(true)
				.setTransposeInput("false");
			CompressionSettings cs = csb.create();

			final CompressedSizeInfoColGroup cgi = new ComEstExact(mbt, cs).getColGroupInfo(colIndexes);
			CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

			assertTrue("Column group should be DeltaDDC, not Const", cg instanceof ColGroupDeltaDDC);

			MatrixBlock ret = new MatrixBlock(numRows, numCols, false);
			ret.allocateDenseBlock();
			cg.decompressToDenseBlock(ret.getDenseBlock(), rl, ru);

			MatrixBlock expected = DataConverter.convertToMatrixBlock(data);
			for(int i = rl; i < ru; i++) {
				for(int j = 0; j < numCols; j++) {
					double expectedValue = expected.get(i, j);
					double actualValue = ret.get(i, j);
					assertArrayEquals(new double[] {expectedValue}, new double[] {actualValue}, 0.01);
				}
			}

		}
		catch(NotImplementedException e) {
			throw e;
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Failed partial range decompression : " + this.getClass().getSimpleName(), e);
		}
	}

	@Test
	public void testSerializationSingleColumn() throws IOException {
		double[][] data = {{1}, {2}, {3}, {4}, {5}};
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC))
			.setPreferDeltaEncoding(true)
			.setTransposeInput("false");
		CompressionSettings cs = csb.create();

		final CompressedSizeInfoColGroup cgi = new ComEstExact(mbt, cs).getDeltaColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		AColGroup original = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

		assertTrue("Original should be ColGroupDeltaDDC", original instanceof ColGroupDeltaDDC);

		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		ColGroupIO.writeGroups(dos, Collections.singletonList(original));
		assertEquals(original.getExactSizeOnDisk() + 4, bos.size());

		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		AColGroup deserialized = ColGroupIO.readGroups(dis, numRows).get(0);

		assertTrue("Deserialized should be ColGroupDeltaDDC", deserialized instanceof ColGroupDeltaDDC);
		assertEquals("Compression type should match", original.getCompType(), deserialized.getCompType());
		assertEquals("Exact size on disk should match", original.getExactSizeOnDisk(), deserialized.getExactSizeOnDisk());

		MatrixBlock originalDecompressed = new MatrixBlock(numRows, numCols, false);
		originalDecompressed.allocateDenseBlock();
		original.decompressToDenseBlock(originalDecompressed.getDenseBlock(), 0, numRows);

		MatrixBlock deserializedDecompressed = new MatrixBlock(numRows, numCols, false);
		deserializedDecompressed.allocateDenseBlock();
		deserialized.decompressToDenseBlock(deserializedDecompressed.getDenseBlock(), 0, numRows);

		for(int i = 0; i < numRows; i++) {
			for(int j = 0; j < numCols; j++) {
				assertArrayEquals(new double[] {originalDecompressed.get(i, j)}, new double[] {deserializedDecompressed.get(i, j)}, 0.01);
			}
		}
	}

	@Test
	public void testSerializationTwoColumns() throws IOException {
		double[][] data = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC))
			.setPreferDeltaEncoding(true)
			.setTransposeInput("false");
		CompressionSettings cs = csb.create();

		final CompressedSizeInfoColGroup cgi = new ComEstExact(mbt, cs).getDeltaColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		AColGroup original = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

		assertTrue("Original should be ColGroupDeltaDDC", original instanceof ColGroupDeltaDDC);

		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		ColGroupIO.writeGroups(dos, Collections.singletonList(original));
		assertEquals(original.getExactSizeOnDisk() + 4, bos.size());

		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		AColGroup deserialized = ColGroupIO.readGroups(dis, numRows).get(0);

		assertTrue("Deserialized should be ColGroupDeltaDDC", deserialized instanceof ColGroupDeltaDDC);
		assertEquals("Compression type should match", original.getCompType(), deserialized.getCompType());
		assertEquals("Exact size on disk should match", original.getExactSizeOnDisk(), deserialized.getExactSizeOnDisk());

		MatrixBlock originalDecompressed = new MatrixBlock(numRows, numCols, false);
		originalDecompressed.allocateDenseBlock();
		original.decompressToDenseBlock(originalDecompressed.getDenseBlock(), 0, numRows);

		MatrixBlock deserializedDecompressed = new MatrixBlock(numRows, numCols, false);
		deserializedDecompressed.allocateDenseBlock();
		deserialized.decompressToDenseBlock(deserializedDecompressed.getDenseBlock(), 0, numRows);

		for(int i = 0; i < numRows; i++) {
			for(int j = 0; j < numCols; j++) {
				assertArrayEquals(new double[] {originalDecompressed.get(i, j)}, new double[] {deserializedDecompressed.get(i, j)}, 0.01);
			}
		}
	}

	@Test
	public void testScalarEquals() {
		double[][] data = {{0}, {1}, {2}, {3}, {0}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(Equals.getEqualsFnObject(), 0.0);
		AColGroup res = cg.scalarOperation(op);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(0.0, ret.get(1, 0), 0.0);
		assertEquals(0.0, ret.get(2, 0), 0.0);
		assertEquals(0.0, ret.get(3, 0), 0.0);
		assertEquals(1.0, ret.get(4, 0), 0.0);
	}
	
	@Test
	public void testScalarGreaterThan() {
		double[][] data = {{0}, {1}, {2}, {3}, {0}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(GreaterThan.getGreaterThanFnObject(), 1.5);
		AColGroup res = cg.scalarOperation(op);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(0.0, ret.get(0, 0), 0.0);
		assertEquals(0.0, ret.get(1, 0), 0.0);
		assertEquals(1.0, ret.get(2, 0), 0.0);
		assertEquals(1.0, ret.get(3, 0), 0.0);
		assertEquals(0.0, ret.get(4, 0), 0.0);
	}

	@Test
	public void testScalarPlus() {
		double[][] data = {{1}, {2}, {3}, {4}, {5}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 10.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain DeltaDDC after shift", res instanceof ColGroupDeltaDDC);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(11.0, ret.get(0, 0), 0.0);
		assertEquals(12.0, ret.get(1, 0), 0.0);
		assertEquals(13.0, ret.get(2, 0), 0.0);
		assertEquals(14.0, ret.get(3, 0), 0.0);
		assertEquals(15.0, ret.get(4, 0), 0.0);
	}

	@Test
	public void testScalarMinus() {
		double[][] data = {{11}, {12}, {13}, {14}, {15}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(Minus.getMinusFnObject(), 10.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain DeltaDDC after shift", res instanceof ColGroupDeltaDDC);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(2.0, ret.get(1, 0), 0.0);
		assertEquals(3.0, ret.get(2, 0), 0.0);
		assertEquals(4.0, ret.get(3, 0), 0.0);
		assertEquals(5.0, ret.get(4, 0), 0.0);
	}

	@Test
	public void testUnaryOperationSqrt() {
		double[][] data = {{1}, {4}, {9}, {16}, {25}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		UnaryOperator op = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.SQRT));
		AColGroup res = cg.unaryOperation(op);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(1.0, ret.get(0, 0), 0.01);
		assertEquals(2.0, ret.get(1, 0), 0.01);
		assertEquals(3.0, ret.get(2, 0), 0.01);
		assertEquals(4.0, ret.get(3, 0), 0.01);
		assertEquals(5.0, ret.get(4, 0), 0.01);
	}

	@Test
	public void testScalarEqualsMultiColumn() {
		double[][] data = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 1}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(Equals.getEqualsFnObject(), 0.0);
		AColGroup res = cg.scalarOperation(op);
		
		MatrixBlock ret = new MatrixBlock(5, 2, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(0.0, ret.get(0, 1), 0.0);
		assertEquals(0.0, ret.get(1, 0), 0.0);
		assertEquals(0.0, ret.get(1, 1), 0.0);
		assertEquals(0.0, ret.get(2, 0), 0.0);
		assertEquals(0.0, ret.get(2, 1), 0.0);
		assertEquals(0.0, ret.get(3, 0), 0.0);
		assertEquals(0.0, ret.get(3, 1), 0.0);
		assertEquals(1.0, ret.get(4, 0), 0.0);
		assertEquals(0.0, ret.get(4, 1), 0.0);
	}

	@Test
	public void testScalarMultiply() {
		double[][] data = {{1}, {2}, {3}, {4}, {5}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(Multiply.getMultiplyFnObject(), 2.0);
		AColGroup res = cg.scalarOperation(op);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(2.0, ret.get(0, 0), 0.0);
		assertEquals(4.0, ret.get(1, 0), 0.0);
		assertEquals(6.0, ret.get(2, 0), 0.0);
		assertEquals(8.0, ret.get(3, 0), 0.0);
		assertEquals(10.0, ret.get(4, 0), 0.0);
	}

	@Test
	public void testScalarDivide() {
		double[][] data = {{2}, {4}, {6}, {8}, {10}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDeltaDDC);
		
		ScalarOperator op = new RightScalarOperator(Divide.getDivideFnObject(), 2.0);
		AColGroup res = cg.scalarOperation(op);
		
		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);
		
		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(2.0, ret.get(1, 0), 0.0);
		assertEquals(3.0, ret.get(2, 0), 0.0);
		assertEquals(4.0, ret.get(3, 0), 0.0);
		assertEquals(5.0, ret.get(4, 0), 0.0);
	}

	@Test
	public void testSliceRows() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
		AColGroup cg = compressForTest(data);
		
		AColGroup sliced = cg.sliceRows(1, 4);
		assertTrue(sliced instanceof ColGroupDeltaDDC);
		
		MatrixBlock ret = new MatrixBlock(3, 2, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);
		
		assertEquals(3.0, ret.get(0, 0), 0.0);
		assertEquals(4.0, ret.get(0, 1), 0.0);
		assertEquals(5.0, ret.get(1, 0), 0.0);
		assertEquals(6.0, ret.get(1, 1), 0.0);
		assertEquals(7.0, ret.get(2, 0), 0.0);
		assertEquals(8.0, ret.get(2, 1), 0.0);
	}

	private AColGroup compressForTest(double[][] data) {
		MatrixBlock mb = DataConverter.convertToMatrixBlock(data);
		IColIndex colIndexes = ColIndexFactory.create(data[0].length);
		CompressionSettings cs = new CompressionSettingsBuilder()
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC))
			.setPreferDeltaEncoding(true)
			.create();
		
		final CompressedSizeInfoColGroup cgi = new ComEstExact(mb, cs).getDeltaColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		return ColGroupFactory.compressColGroups(mb, csi, cs, 1).get(0);
	}

}
