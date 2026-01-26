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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDCLZW;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupIO;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.estim.ComEstExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class ColGroupDDCLZWTest {
	protected static final Log LOG = LogFactory.getLog(ColGroupDDCLZWTest.class.getName());

	public void testConvertToDDCLZW() {
	}

	@Test
	public void testConvertToDDCLZWBasic() {
		IColIndex colIndexes = ColIndexFactory.create(2);
		double[] dictValues = new double[] {10.0, 20.0, 11.0, 21.0, 12.0, 22.0};
		Dictionary dict = Dictionary.create(dictValues);
		AMapToData data = MapToFactory.create(3, 3);
		data.set(0, 0);
		data.set(1, 1);
		data.set(2, 2);

		ColGroupDDC ddc = (ColGroupDDC) ColGroupDDC.create(colIndexes, dict, data, null);
		AColGroup result = ddc.convertToDDCLZW();

		assertNotNull(result);
		assertTrue(result instanceof ColGroupDDCLZW);
		ColGroupDDCLZW DDCLZW = (ColGroupDDCLZW) result;

		MatrixBlock mb = new MatrixBlock(3, 2, false);
		mb.allocateDenseBlock();
		DDCLZW.decompressToDenseBlock(mb.getDenseBlock(), 0, 3);

		assertEquals(10.0, mb.get(0, 0), 0.0);
		assertEquals(20.0, mb.get(0, 1), 0.0);
		assertEquals(11.0, mb.get(1, 0), 0.0);
		assertEquals(21.0, mb.get(1, 1), 0.0);
		assertEquals(12.0, mb.get(2, 0), 0.0);
		assertEquals(22.0, mb.get(2, 1), 0.0);
	}

	/**
	 * Creates a sample DDC group for unit tests
	 */
	private ColGroupDDC createTestDDC(int[] mapping, int nCols, int nUnique) {
		IColIndex colIndexes = ColIndexFactory.create(nCols);

		double[] dictValues = new double[nUnique * nCols];
		for(int i = 0; i < nUnique; i++) {
			for(int c = 0; c < nCols; c++) {
				dictValues[i * nCols + c] = (i + 1) * 10.0 + c;
			}
		}
		Dictionary dict = Dictionary.create(dictValues);

		AMapToData data = MapToFactory.create(mapping.length, nUnique);
		for(int i = 0; i < mapping.length; i++) {
			data.set(i, mapping[i]);
		}

		AColGroup result = ColGroupDDC.create(colIndexes, dict, data, null);
		assertTrue("The result is of class '" + result.getClass() + "'", result instanceof ColGroupDDC);
		return (ColGroupDDC) result;
	}

	/**
	 * Asserts that two maps are identical
	 */
	private void assertMapsEqual(AMapToData expected, AMapToData actual) {
		assertEquals("Size mismatch", expected.size(), actual.size());
		assertEquals("Unique count mismatch", expected.getUnique(), actual.getUnique());

		for(int i = 0; i < expected.size(); i++) {
			assertEquals("Mapping mismatch at row " + i, expected.getIndex(i), actual.getIndex(i));
		}
	}

	/**
	 * Applies DDCLZW compression/decompression and asserts that it's left unchanged
	 */
	private void assertLosslessCompression(ColGroupDDC original) {
		// Compress
		AColGroup compressed = original.convertToDDCLZW();
		assertNotNull("Compression returned null", compressed);
		assertTrue(compressed instanceof ColGroupDDCLZW);

		// Decompress
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) compressed;
		AColGroup decompressed = ddclzw.convertToDDC();
		assertNotNull("Decompression returned null", decompressed);
		assertTrue(decompressed instanceof ColGroupDDC);

		// Assert
		ColGroupDDC result = (ColGroupDDC) decompressed;

		AMapToData d1 = original.getMapToData();
		AMapToData d2 = result.getMapToData();

		assertMapsEqual(d1, d2);
		assertEquals("Column indices mismatch", original.getColIndices(), result.getColIndices());

		assertEquals("Size mismatch", d1.size(), d2.size());
		assertEquals("Unique count mismatch", d1.getUnique(), d2.getUnique());

		for(int i = 0; i < d1.size(); i++) {
			assertEquals("Mapping mismatch at row " + i, d1.getIndex(i), d2.getIndex(i));
		}
	}

	/**
	 * Asserts "partial decompression" up to the `index`
	 */
	private void assertPartialDecompression(ColGroupDDCLZW ddclzw, AMapToData original, int index) {
		ColGroupDDC partial = (ColGroupDDC) ddclzw.convertToDDC();
		AMapToData partialMap = partial.getMapToData();

		assertEquals("Partial size incorrect", index, partialMap.size());

		for(int i = 0; i < index; i++) {
			assertEquals("Partial map mismatch at " + i, original.getIndex(i), partialMap.getIndex(i));
		}
	}

	/**
	 * Asserts if the slice operation matches DDC's slice
	 */
	private void assertSlice(ColGroupDDCLZW ddclzw, ColGroupDDC originalDDC, int low, int high) {
		AColGroup sliced = ddclzw.sliceRows(low, high);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		ColGroupDDCLZW ddclzwSlice = (ColGroupDDCLZW) sliced;
		ColGroupDDC ddcSlice = (ColGroupDDC) ddclzwSlice.convertToDDC();
		ColGroupDDC expectedSlice = (ColGroupDDC) originalDDC.sliceRows(low, high);

		assertMapsEqual(expectedSlice.getMapToData(), ddcSlice.getMapToData());
	}

	@Test
	public void testConvertToDDCLZWBasicNew() {
		int[] src = new int[] {0, 0, 2, 0, 2, 1, 0, 2, 1, 0, 2, 2, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 1, 1,
			0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 0, 2, 1, 0, 2, 1, 0, 0, 2, 1, 0,
			2, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 1, 0, 2, 1, 0, 2, 2, 0, 2, 1, 0, 2, 1, 0, 2, 0, 0, 0, 0, 0, 1};

		// Create DDC with 2 columns, 3 unique values
		ColGroupDDC ddc = createTestDDC(src, 2, 3);

		assertLosslessCompression(ddc);

		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();
		assertPartialDecompression(ddclzw, ddc.getMapToData(), 101);
		assertSlice(ddclzw, ddc, 3, 10);
	}

	@Test(expected = IllegalArgumentException.class)
	public void testPartialDecompressionOutOfBounds() {
		int[] src = new int[] {1, 3, 4, 4, 3, 2, 3, 4, 1, 4, 4, 4, 4, 1, 4, 1, 4, 1, 4, 0, 1, 3, 4, 4, 3, 2, 3, 4, 1, 4,
			4, 4, 4, 1, 4, 1, 4, 1, 4, 0,};

		ColGroupDDC ddc = createTestDDC(src, 3, 5);

		assertLosslessCompression(ddc);

		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();
		assertPartialDecompression(ddclzw, ddc.getMapToData(), 40);
		/*assertPartialDecompression(ddclzw, ddc.getMapToData(), 41); // Should throw out of bounds*/
	}

	/*@Test
	public void testLengthTwo() {
		int[] src = new int[] {0, 1};

		ColGroupDDC ddc = createTestDDC(src, 1, 2);

		assertLosslessCompression(ddc);

		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();
		assertPartialDecompression(ddclzw, ddc.getMapToData(), 0);
		assertPartialDecompression(ddclzw, ddc.getMapToData(), 2);
	}*/

	@Test
	public void testGetIdxFirstElement() {
		int[] src = new int[] {0, 1, 2, 1, 0};
		ColGroupDDC ddc = createTestDDC(src, 2, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		double expected = ddc.getIdx(0, 0);
		assertEquals(expected, ddclzw.getIdx(0, 0), 0.0001);
	}

	@Test
	public void testGetIdxLastElement() {
		int[] src = new int[] {0, 1, 2, 1, 0};
		ColGroupDDC ddc = createTestDDC(src, 2, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		int lastRow = src.length - 1;
		double expected = ddc.getIdx(lastRow, 1);
		assertEquals(expected, ddclzw.getIdx(lastRow, 1), 0.0001);
	}

	@Test
	public void testGetIdxAllElements() {
		int[] src = new int[] {0, 1, 2, 1, 0, 2, 1};
		ColGroupDDC ddc = createTestDDC(src, 3, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		for(int row = 0; row < src.length; row++) {
			for(int col = 0; col < 2; col++) {
				double expected = ddc.getIdx(row, col);
				double actual = ddclzw.getIdx(row, col);
				assertEquals("Mismatch at [" + row + "," + col + "]", expected, actual, 0.0001);
			}
		}
	}

	@Test
	public void testGetIdxWithRepeatingPattern() {
		int[] src = new int[] {0, 1, 0, 1, 0, 1, 0, 1};
		ColGroupDDC ddc = createTestDDC(src, 1, 2);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		double expected = ddc.getIdx(3, 0);
		assertEquals(expected, ddclzw.getIdx(3, 0), 0.0001);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testGetIdxRowOutOfBoundsNegative() {
		int[] src = new int[] {0, 1, 2};
		ColGroupDDC ddc = createTestDDC(src, 1, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		ddclzw.getIdx(-1, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testGetIdxRowOutOfBounds() {
		int[] src = new int[] {0, 1, 2};
		ColGroupDDC ddc = createTestDDC(src, 1, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		ddclzw.getIdx(10, 0);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testGetIdxColOutOfBoundsNegative() {
		int[] src = new int[] {0, 1, 2};
		ColGroupDDC ddc = createTestDDC(src, 3, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		ddclzw.getIdx(0, -1);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testGetIdxColOutOfBounds() {
		int[] src = new int[] {0, 1, 2};
		ColGroupDDC ddc = createTestDDC(src, 3, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		ddclzw.getIdx(0, 10);
	}

	@Test
	public void testCreateWithNullDictionary() {
		IColIndex colIndexes = ColIndexFactory.create(1);
		int[] src = new int[] {0, 1, 2};
		AMapToData data = MapToFactory.create(3, 3);
		for(int i = 0; i < 3; i++) {
			data.set(i, src[i]);
		}

		AColGroup result = ColGroupDDCLZW.create(colIndexes, null, data, null);
		assertTrue("Should create ColGroupEmpty", result instanceof ColGroupEmpty);
	}

	@Test
	public void testCreateWithSingleUnique() {
		IColIndex colIndexes = ColIndexFactory.create(1);
		double[] dictValues = new double[] {42.0};
		Dictionary dict = Dictionary.create(dictValues);

		int[] src = new int[] {0, 0, 0, 0};
		AMapToData data = MapToFactory.create(4, 1);
		for(int i = 0; i < 4; i++) {
			data.set(i, 0);
		}

		AColGroup result = ColGroupDDCLZW.create(colIndexes, dict, data, null);
		assertTrue("Should create ColGroupConst", result instanceof ColGroupConst);
	}

	@Test
	public void testCreateValidDDCLZW() {
		int[] src = new int[] {0, 1, 0, 1, 2};
		ColGroupDDC ddc = createTestDDC(src, 1, 3);

		AColGroup result = ddc.convertToDDCLZW();
		assertTrue("Should create ColGroupDDCLZW", result instanceof ColGroupDDCLZW);
	}

	@Test
	public void testCreateWithMultipleColumns() {
		int[] src = new int[] {0, 1, 2, 1, 0};
		ColGroupDDC ddc = createTestDDC(src, 3, 3);

		AColGroup result = ddc.convertToDDCLZW();
		assertTrue("Should create ColGroupDDCLZW", result instanceof ColGroupDDCLZW);
	}

	@Test
	public void testSameNumber() {
		int[] src = new int[20];
		Arrays.fill(src, 2);

		ColGroupDDC ddc = createTestDDC(src, 1, 3);
		assertLosslessCompression(ddc);
	}

	@Test
	public void testAlternatingNumbers() {
		int[] src = new int[30];
		for(int i = 0; i < src.length; i++) {
			src[i] = i % 2;
		}

		ColGroupDDC ddc = createTestDDC(src, 1, 2);
		assertLosslessCompression(ddc);
	}

	@Test
	public void testLongPatterns() {
		int[] src = new int[50];
		Arrays.fill(src, 0, 15, 0);
		Arrays.fill(src, 15, 30, 1);
		Arrays.fill(src, 30, 45, 2);
		Arrays.fill(src, 45, 50, 0);

		ColGroupDDC ddc = createTestDDC(src, 1, 3);
		assertLosslessCompression(ddc);
	}

	@Test
	public void testSameIndexStructure() {
		int[] src = new int[] {0, 1, 0, 1};
		ColGroupDDC ddc = createTestDDC(src, 1, 2);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		assertTrue("Same object should have same structure", ddclzw.sameIndexStructure(ddclzw));
	}

	@Test
	public void testSameIndexStructureDifferent() {
		int[] src = new int[] {0, 1, 0, 1};

		ColGroupDDC ddc1 = createTestDDC(src, 1, 2);
		ColGroupDDC ddc2 = createTestDDC(src, 1, 2);

		ColGroupDDCLZW ddclzw1 = (ColGroupDDCLZW) ddc1.convertToDDCLZW();
		ColGroupDDCLZW ddclzw2 = (ColGroupDDCLZW) ddc2.convertToDDCLZW();

		// Different objects have different _dataLZW arrays
		assertFalse("Different objects should have different structure", ddclzw1.sameIndexStructure(ddclzw2));
	}

	@Test
	public void testSameIndexStructureDdcLzw() {
		int[] src = new int[] {0, 1, 2, 1, 0};
		ColGroupDDC ddc = createTestDDC(src, 1, 3);
		ColGroupDDCLZW ddclzw = (ColGroupDDCLZW) ddc.convertToDDCLZW();

		assertFalse("Different types should not have same structure", ddclzw.sameIndexStructure(ddc));
	}

	@Test
	public void testRepetitiveData() {
		int[] src = new int[] {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
			1};

		ColGroupDDC ddc = createTestDDC(src, 1, 2);
		assertLosslessCompression(ddc);
	}

	public void assertLosslessCompression_NoRepetition(ColGroupDDCLZW original, int nRows) {
		AColGroup decompressed = original.convertToDDC();
		assertNotNull(decompressed);
		assertTrue(decompressed instanceof ColGroupDDC);

		ColGroupDDC result = (ColGroupDDC) decompressed;

		AMapToData map = result.getMapToData();
		assertNotNull(map);

		assertEquals(nRows, map.size());
		assertEquals(nRows, map.getUnique());

		for(int i = 0; i < nRows; i++) {
			assertEquals("Mapping mismatch at row " + i, i, map.getIndex(i));
		}
	}

	@Test
	public void testNoRepetition() {
		double[][] data = new double[20][1];
		for(int i = 0; i < 20; i++) {
			data[i][0] = i;
		}

		AColGroup cg = compressForTest(data);
		assertNotNull(cg);
		assertTrue(cg instanceof ColGroupDDCLZW);

		assertLosslessCompression_NoRepetition((ColGroupDDCLZW) cg, 20);
	}

	public void testDecompressToDenseBlock(double[][] data, boolean isTransposed) {
		if(isTransposed) {
			throw new NotImplementedException("ColGroup enLZWcoding for transposed matrices not yet implemented");
		}

		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		try {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DDCLZW)).setTransposeInput("false");
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

	public void testDecompressToDenseBlockPartialRange(double[][] data, boolean isTransposed, int rl, int ru) {
		if(isTransposed) {
			throw new NotImplementedException("ColGroup enLZWcoding for transposed matrices not yet implemented");
		}

		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		try {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DDCLZW)).setTransposeInput("false");
			CompressionSettings cs = csb.create();

			final CompressedSizeInfoColGroup cgi = new ComEstExact(mbt, cs).getColGroupInfo(colIndexes);
			CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

			assertTrue("Column group should be DDCLZW, not Const", cg instanceof ColGroupDDCLZW);

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
	public void testDecompressToDenseBlockPartialRangeSingleColumn() {
		testDecompressToDenseBlockPartialRange(new double[][] {{1}, {2}, {3}, {4}, {5}}, false, 2, 5);
	}

	@Test
	public void testDecompressToDenseBlockPartialRangeTwoColumns() {
		testDecompressToDenseBlockPartialRange(new double[][] {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}}, false, 1, 4);
	}

	@Test
	public void testDecompressToDenseBlockPartialRangeFromMiddle() {
		testDecompressToDenseBlockPartialRange(new double[][] {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}}, false,
			3, 6);
	}

	@Test
	public void testSerializationSingleColumn() throws IOException {
		double[][] data = {{1}, {2}, {3}, {4}, {5}};
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);
		final int numCols = mbt.getNumColumns();
		final int numRows = mbt.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(numCols);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DDCLZW)).setTransposeInput("false");
		CompressionSettings cs = csb.create();

		final CompressedSizeInfoColGroup cgi = new ComEstExact(mbt, cs).getColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

		assertTrue("Original should be ColGroupDDCLZW", cg instanceof ColGroupDDCLZW);

		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		ColGroupIO.writeGroups(dos, Collections.singletonList(cg));
		assertEquals(cg.getExactSizeOnDisk() + 4, bos.size());

		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		AColGroup deserialized = ColGroupIO.readGroups(dis, numRows).get(0);

		assertTrue("Deserialized should be ColGroupDDCLZW", deserialized instanceof ColGroupDDCLZW);
		assertEquals("Compression type should match", cg.getCompType(), deserialized.getCompType());
		assertEquals("Exact size on disk should match", cg.getExactSizeOnDisk(), deserialized.getExactSizeOnDisk());

		MatrixBlock originalDecompressed = new MatrixBlock(numRows, numCols, false);
		originalDecompressed.allocateDenseBlock();
		cg.decompressToDenseBlock(originalDecompressed.getDenseBlock(), 0, numRows);

		MatrixBlock deserializedDecompressed = new MatrixBlock(numRows, numCols, false);
		deserializedDecompressed.allocateDenseBlock();
		deserialized.decompressToDenseBlock(deserializedDecompressed.getDenseBlock(), 0, numRows);

		for(int i = 0; i < numRows; i++) {
			for(int j = 0; j < numCols; j++) {
				assertArrayEquals(new double[] {originalDecompressed.get(i, j)},
					new double[] {deserializedDecompressed.get(i, j)}, 0.01);
			}
		}
	}

	@Test
	public void testSerializationTwoColumns() throws IOException {
		double[][] data = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};

		MatrixBlock mb = DataConverter.convertToMatrixBlock(data);
		final int numCols = mb.getNumColumns();
		final int numRows = mb.getNumRows();
		IColIndex colIndexes = ColIndexFactory.create(data[0].length);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DDCLZW)).setTransposeInput("false");
		CompressionSettings cs = csb.create();

		final CompressedSizeInfoColGroup cgi = new ComEstExact(mb, cs).getColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		AColGroup original = ColGroupFactory.compressColGroups(mb, csi, cs, 1).get(0);

		assertTrue("Original should be ColgroupDDCLZW", original instanceof ColGroupDDCLZW);

		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(bos);
		ColGroupIO.writeGroups(dos, Collections.singletonList(original));
		assertEquals(original.getExactSizeOnDisk() + 4, bos.size());

		ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
		DataInputStream dis = new DataInputStream(bis);
		AColGroup deserialized = ColGroupIO.readGroups(dis, numRows).get(0);

		assertTrue("Deserialized should be ColGroupDDCLZW", deserialized instanceof ColGroupDDCLZW);
		assertEquals("Compression type should match", original.getCompType(), deserialized.getCompType());
		assertEquals("Exact size on disk should match", original.getExactSizeOnDisk(),
			deserialized.getExactSizeOnDisk());

		MatrixBlock originalDecompressed = new MatrixBlock(numRows, numCols, false);
		originalDecompressed.allocateDenseBlock();
		original.decompressToDenseBlock(originalDecompressed.getDenseBlock(), 0, numRows);

		MatrixBlock deserializedDecompressed = new MatrixBlock(numRows, numCols, false);
		deserializedDecompressed.allocateDenseBlock();
		deserialized.decompressToDenseBlock(deserializedDecompressed.getDenseBlock(), 0, numRows);

		for(int i = 0; i < numRows; i++) {
			for(int j = 0; j < numCols; j++) {
				assertArrayEquals(new double[] {originalDecompressed.get(i, j)},
					new double[] {deserializedDecompressed.get(i, j)}, 0.01);
			}
		}
	}

	private AColGroup compressForTest(double[][] data) {
		MatrixBlock mb = DataConverter.convertToMatrixBlock(data);
		IColIndex colIndexes = ColIndexFactory.create(data[0].length);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DDCLZW)).setTransposeInput("false");
		CompressionSettings cs = csb.create();

		final CompressedSizeInfoColGroup cgi = new ComEstExact(mb, cs).getColGroupInfo(colIndexes);
		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
		return ColGroupFactory.compressColGroups(mb, csi, cs, 1).get(0);
	}

	@Test
	public void testScalarEquals() {
		double[][] data = {{0}, {1}, {2}, {3}, {0}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

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
		assertTrue(cg instanceof ColGroupDDCLZW);

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
		assertTrue(cg instanceof ColGroupDDCLZW);

		ScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 10.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain ColGroupDDCLZW after shift", res instanceof ColGroupDDCLZW);

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
		assertTrue(cg instanceof ColGroupDDCLZW);

		ScalarOperator op = new RightScalarOperator(Minus.getMinusFnObject(), 10.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain ColGroupDDCLZW after shift", res instanceof ColGroupDDCLZW);

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
		assertTrue(cg instanceof ColGroupDDCLZW);

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
		assertTrue(cg instanceof ColGroupDDCLZW);

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
		assertTrue(cg instanceof ColGroupDDCLZW);

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

		assertTrue(cg instanceof ColGroupDDCLZW);

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
	public void testSliceRowsSingleRow() {
		double[][] data = {{0}, {1}, {2}, {1}, {0}, {2}, {1}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		AColGroup sliced = cg.sliceRows(3, 4);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(1, 1, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 1);

		assertEquals(1.0, ret.get(0, 0), 0.0);
	}

	@Test
	public void testSliceRowsMiddleRange() {
		double[][] data = {{0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		AColGroup sliced = cg.sliceRows(2, 7);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(5, 1, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 5);

		for(int i = 0; i < 5; i++) {
			assertEquals(data[2 + i][0], ret.get(i, 0), 0.0);
		}
	}

	@Test
	public void testSliceRowsEntireRange() {
		double[][] data = {{0}, {1}, {0}, {1}, {2}};

		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		AColGroup sliced = cg.sliceRows(0, data.length);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(data.length, 1, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, data.length);

		for(int i = 0; i < data.length; i++) {
			assertEquals(data[i][0], ret.get(i, 0), 0.0);
		}
	}

	@Test
	public void testSliceRowsBeginning() {
		double[][] data = {{0}, {1}, {2}, {1}, {0}, {2}};

		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		AColGroup sliced = cg.sliceRows(0, 3);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(3, 1, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(0.0, ret.get(0, 0), 0.0);
		assertEquals(1.0, ret.get(1, 0), 0.0);
		assertEquals(2.0, ret.get(2, 0), 0.0);
	}

	@Test
	public void testSliceRowsEnd() {
		double[][] data = {{0}, {1}, {2}, {1}, {0}, {2}};

		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		AColGroup sliced = cg.sliceRows(3, 6);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(3, 1, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(0.0, ret.get(1, 0), 0.0);
		assertEquals(2.0, ret.get(2, 0), 0.0);
	}

	@Test
	public void testSliceRowsWithLongRuns() {
		double[][] data = new double[30][1];
		Arrays.fill(data, 0, 10, new double[] {0});
		Arrays.fill(data, 10, 20, new double[] {1});
		Arrays.fill(data, 20, 30, new double[] {2});

		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		AColGroup sliced = cg.sliceRows(5, 25);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(20, 1, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 20);

		for(int i = 0; i < 20; i++) {
			double expected = data[i + 5][0];
			assertEquals(expected, ret.get(i, 0), 0.0);
		}
	}

	@Test
	public void testSliceRows() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
		AColGroup cg = compressForTest(data);

		AColGroup sliced = cg.sliceRows(1, 4);
		assertTrue(sliced instanceof ColGroupDDCLZW);

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

	@Test
	public void testSliceRowsWithMatchingDictionaryEntry() {
		double[][] data = {{1, 2}, {3, 4}, {1, 2}, {5, 6}, {7, 8}};
		AColGroup cg = compressForTest(data);

		AColGroup sliced = cg.sliceRows(2, 5);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(3, 2, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(2.0, ret.get(0, 1), 0.0);
		assertEquals(5.0, ret.get(1, 0), 0.0);
		assertEquals(6.0, ret.get(1, 1), 0.0);
		assertEquals(7.0, ret.get(2, 0), 0.0);
		assertEquals(8.0, ret.get(2, 1), 0.0);
	}

	@Test
	public void testSliceRowsWithNoMatchingDictionaryEntry() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}};
		AColGroup cg = compressForTest(data);

		AColGroup sliced = cg.sliceRows(1, 3);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(2, 2, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 2);

		assertEquals(3.0, ret.get(0, 0), 0.0);
		assertEquals(4.0, ret.get(0, 1), 0.0);
		assertEquals(5.0, ret.get(1, 0), 0.0);
		assertEquals(6.0, ret.get(1, 1), 0.0);
	}

	@Test
	public void testSliceRowsFromMiddleRow() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
		AColGroup cg = compressForTest(data);

		AColGroup sliced = cg.sliceRows(2, 4);
		assertTrue(sliced instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(2, 2, false);
		ret.allocateDenseBlock();
		sliced.decompressToDenseBlock(ret.getDenseBlock(), 0, 2);

		assertEquals(5.0, ret.get(0, 0), 0.0);
		assertEquals(6.0, ret.get(0, 1), 0.0);
		assertEquals(7.0, ret.get(1, 0), 0.0);
		assertEquals(8.0, ret.get(1, 1), 0.0);
	}

	@Test
	public void testDecompressToSparseBlock() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}};
		AColGroup cg = compressForTest(data);

		MatrixBlock ret = new MatrixBlock(3, 2, true);
		ret.allocateSparseRowsBlock();
		cg.decompressToSparseBlock(ret.getSparseBlock(), 0, 3);

		assertEquals(1.0, ret.get(0, 0), 0.0);
		assertEquals(2.0, ret.get(0, 1), 0.0);
		assertEquals(3.0, ret.get(1, 0), 0.0);
		assertEquals(4.0, ret.get(1, 1), 0.0);
		assertEquals(5.0, ret.get(2, 0), 0.0);
		assertEquals(6.0, ret.get(2, 1), 0.0);
	}

	@Test
	public void testDecompressToSparseBlockWithRlGreaterThanZero() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
		AColGroup cg = compressForTest(data);

		MatrixBlock ret = new MatrixBlock(4, 2, true);
		ret.allocateSparseRowsBlock();
		cg.decompressToSparseBlock(ret.getSparseBlock(), 2, 4, 0, 0);

		assertEquals(5.0, ret.get(2, 0), 0.0);
		assertEquals(6.0, ret.get(2, 1), 0.0);
		assertEquals(7.0, ret.get(3, 0), 0.0);
		assertEquals(8.0, ret.get(3, 1), 0.0);
	}

	@Test
	public void testDecompressToSparseBlockWithOffset() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}};
		AColGroup cg = compressForTest(data);

		MatrixBlock ret = new MatrixBlock(5, 4, true);
		ret.allocateSparseRowsBlock();
		cg.decompressToSparseBlock(ret.getSparseBlock(), 0, 3, 1, 1);

		assertEquals(1.0, ret.get(1, 1), 0.0);
		assertEquals(2.0, ret.get(1, 2), 0.0);
		assertEquals(3.0, ret.get(2, 1), 0.0);
		assertEquals(4.0, ret.get(2, 2), 0.0);
		assertEquals(5.0, ret.get(3, 1), 0.0);
		assertEquals(6.0, ret.get(3, 2), 0.0);
	}

	@Test
	public void testGetNumberNonZeros() {
		double[][] data = {{1, 0}, {2, 3}, {0, 4}, {5, 0}};
		AColGroup cg = compressForTest(data);

		long nnz = cg.getNumberNonZeros(4);
		assertEquals(5L, nnz);
	}

	@Test
	public void testGetNumberNonZerosAllZeros() {
		double[][] data = {{0, 0}, {0, 0}, {0, 0}};
		AColGroup cg = compressForTest(data);

		long nnz = cg.getNumberNonZeros(3);
		assertEquals(0L, nnz);
	}

	@Test
	public void testGetNumberNonZerosAllNonZeros() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}};
		AColGroup cg = compressForTest(data);

		long nnz = cg.getNumberNonZeros(3);
		assertEquals(6L, nnz);
	}

	@Test
	public void testDecompressToDenseBlockNonContiguousPath() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}};
		AColGroup cg = compressForTest(data);

		MatrixBlock ret = new MatrixBlock(3, 5, false);
		ret.allocateDenseBlock();
		cg.decompressToDenseBlock(ret.getDenseBlock(), 0, 3, 0, 2);

		assertEquals(1.0, ret.get(0, 2), 0.0);
		assertEquals(2.0, ret.get(0, 3), 0.0);
		assertEquals(3.0, ret.get(1, 2), 0.0);
		assertEquals(4.0, ret.get(1, 3), 0.0);
		assertEquals(5.0, ret.get(2, 2), 0.0);
		assertEquals(6.0, ret.get(2, 3), 0.0);
	}

	@Test
	public void testDecompressToDenseBlockFirstRowPath() {
		double[][] data = {{10, 20}, {11, 21}, {12, 22}};
		AColGroup cg = compressForTest(data);

		MatrixBlock ret = new MatrixBlock(3, 2, false);
		ret.allocateDenseBlock();
		cg.decompressToDenseBlock(ret.getDenseBlock(), 0, 1);

		assertEquals(10.0, ret.get(0, 0), 0.0);
		assertEquals(20.0, ret.get(0, 1), 0.0);
	}

	@Test
	public void testScalarOperationShiftWithExistingMatch() {
		double[][] data = {{1}, {2}, {3}, {1}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		ScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 1.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain DeltaDDC after shift", res instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(4, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 4);

		assertEquals(2.0, ret.get(0, 0), 0.0);
		assertEquals(3.0, ret.get(1, 0), 0.0);
		assertEquals(4.0, ret.get(2, 0), 0.0);
		assertEquals(2.0, ret.get(3, 0), 0.0);
	}

	@Test
	public void testScalarOperationShiftWithCountsId0EqualsOne() {
		double[][] data = {{1}, {2}, {3}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		ScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 5.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain DeltaDDC after shift", res instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(3, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(6.0, ret.get(0, 0), 0.0);
		assertEquals(7.0, ret.get(1, 0), 0.0);
		assertEquals(8.0, ret.get(2, 0), 0.0);
	}

	@Test
	public void testScalarOperationShiftWithNoMatch() {
		double[][] data = {{1}, {2}, {3}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		ScalarOperator op = new RightScalarOperator(Plus.getPlusFnObject(), 10.0);
		AColGroup res = cg.scalarOperation(op);
		assertTrue("Should remain DeltaDDC after shift", res instanceof ColGroupDDCLZW);

		MatrixBlock ret = new MatrixBlock(3, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(11.0, ret.get(0, 0), 0.0);
		assertEquals(12.0, ret.get(1, 0), 0.0);
		assertEquals(13.0, ret.get(2, 0), 0.0);
	}

	@Test
	public void testUnaryOperationTriggersConvertToDDC() {
		double[][] data = {{1, 2}, {3, 4}, {5, 6}};
		AColGroup cg = compressForTest(data);
		assertTrue(cg instanceof ColGroupDDCLZW);

		UnaryOperator op = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.ABS));
		AColGroup res = cg.unaryOperation(op);

		MatrixBlock ret = new MatrixBlock(3, 2, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(1.0, ret.get(0, 0), 0.01);
		assertEquals(2.0, ret.get(0, 1), 0.01);
		assertEquals(3.0, ret.get(1, 0), 0.01);
		assertEquals(4.0, ret.get(1, 1), 0.01);
		assertEquals(5.0, ret.get(2, 0), 0.01);
		assertEquals(6.0, ret.get(2, 1), 0.01);
	}

	@Test
	public void testUnaryOperationWithConstantResultSingleColumn() {
		double[][] data = {{5}, {5}, {5}, {5}};
		AColGroup cg = compressForTest(data);
		/*assertTrue(cg instanceof ColGroupDDCLZW);*/ // Type CONST.

		UnaryOperator op = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.ABS));
		AColGroup res = cg.unaryOperation(op);

		MatrixBlock ret = new MatrixBlock(4, 1, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 4);

		assertEquals(5.0, ret.get(0, 0), 0.01);
		assertEquals(5.0, ret.get(1, 0), 0.01);
		assertEquals(5.0, ret.get(2, 0), 0.01);
		assertEquals(5.0, ret.get(3, 0), 0.01);
	}

	@Test
	public void testUnaryOperationWithConstantResultMultiColumn() {
		double[][] data = {{10, 20}, {10, 20}, {10, 20}};
		AColGroup cg = compressForTest(data);
		/*assertTrue(cg instanceof ColGroupDDCLZW);*/

		UnaryOperator op = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.ABS));
		AColGroup res = cg.unaryOperation(op);

		MatrixBlock ret = new MatrixBlock(3, 2, false);
		ret.allocateDenseBlock();
		res.decompressToDenseBlock(ret.getDenseBlock(), 0, 3);

		assertEquals(10.0, ret.get(0, 0), 0.01);
		assertEquals(20.0, ret.get(0, 1), 0.01);
		assertEquals(10.0, ret.get(1, 0), 0.01);
		assertEquals(20.0, ret.get(1, 1), 0.01);
		assertEquals(10.0, ret.get(2, 0), 0.01);
		assertEquals(20.0, ret.get(2, 1), 0.01);
	}

	private static MatrixBlock decompressToMB(AColGroup g, int rows, int cols) {
		MatrixBlock ret = new MatrixBlock(rows, cols, false);
		ret.allocateDenseBlock();
		g.decompressToDenseBlock(ret.getDenseBlock(), 0, rows);
		return ret;
	}

	private static void assertMatrixEquals(double[][] expected, MatrixBlock actual) {
		assertEquals(expected.length, actual.getNumRows());
		assertEquals(expected[0].length, actual.getNumColumns());

		for(int r = 0; r < expected.length; r++) {
			for(int c = 0; c < expected[0].length; c++) {
				assertEquals("Mismatch at (" + r + "," + c + ")", expected[r][c], actual.get(r, c), 0.0);
			}
		}
	}

	@Test
	public void testConvertToDDCRoundtripEqualsOriginalData() {
		double[][] data = new double[][] {{1, 2}, {3, 4}, {1, 2}, {5, 6}, {1, 2}, {3, 4}, {7, 8}, {1, 2}, {5, 6},
			{1, 2},};

		AColGroup cg = compressForTest(data);
		assertTrue("Expected DDCLZW from compression framework but was " + cg.getClass().getSimpleName(),
			cg instanceof ColGroupDDCLZW);

		AColGroup ddc = ((ColGroupDDCLZW) cg).convertToDDC();
		assertTrue("Expected ColGroupDDC but was " + ddc.getClass().getSimpleName(), ddc instanceof ColGroupDDC);

		MatrixBlock mbLZW = decompressToMB(cg, data.length, data[0].length);
		MatrixBlock mbDDC = decompressToMB(ddc, data.length, data[0].length);

		assertMatrixEquals(data, mbLZW);
		assertMatrixEquals(data, mbDDC);
	}
}
