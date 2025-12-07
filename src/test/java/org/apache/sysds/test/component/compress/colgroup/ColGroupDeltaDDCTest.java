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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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

}
