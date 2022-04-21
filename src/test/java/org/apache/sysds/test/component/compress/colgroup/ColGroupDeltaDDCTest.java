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

public class ColGroupDeltaDDCTest {

	// protected static final Log LOG = LogFactory.getLog(JolEstimateTest.class.getName());

	// @Test
	// public void testDecompressToDenseBlockSingleColumn() {
	// 	testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}}, true);
	// }

	// @Test
	// public void testDecompressToDenseBlockSingleColumnTransposed() {
	// 	testDecompressToDenseBlock(new double[][] {{1}, {2}, {3}, {4}, {5}}, false);
	// }

	// @Test
	// public void testDecompressToDenseBlockTwoColumns() {
	// 	testDecompressToDenseBlock(new double[][] {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}, false);
	// }

	// @Test
	// public void testDecompressToDenseBlockTwoColumnsTransposed() {
	// 	testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}, {1, 1, 1, 1, 1}}, true);
	// }

	// public void testDecompressToDenseBlock(double[][] data, boolean isTransposed) {
	// 	MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

	// 	final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
	// 	final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
	// 	int[] colIndexes = new int[numCols];
	// 	for(int x = 0; x < numCols; x++)
	// 		colIndexes[x] = x;

	// 	try {
	// 		CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
	// 			.setValidCompressions(EnumSet.of(AColGroup.CompressionType.DeltaDDC)).create();
	// 		cs.transposed = isTransposed;

	// 		final CompressedSizeInfoColGroup cgi = new CompressedSizeEstimatorExact(mbt, cs)
	// 			.getColGroupInfo(colIndexes);
	// 		CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
	// 		AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

	// 		// Decompress to dense block
	// 		MatrixBlock ret = new MatrixBlock(numRows, numCols, false);
	// 		ret.allocateDenseBlock();
	// 		cg.decompressToDenseBlock(ret.getDenseBlock(), 0, numRows);

	// 		MatrixBlock expected = DataConverter.convertToMatrixBlock(data);
	// 		if(isTransposed)
	// 			LibMatrixReorg.transposeInPlace(expected, 1);
	// 		Assert.assertArrayEquals(expected.getDenseBlockValues(), ret.getDenseBlockValues(), 0.01);

	// 	}
	// 	catch(Exception e) {
	// 		e.printStackTrace();
	// 		throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName());
	// 	}
	// }

}
