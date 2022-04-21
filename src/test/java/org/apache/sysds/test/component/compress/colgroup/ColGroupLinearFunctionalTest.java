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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.junit.Assert;
import org.junit.Test;

import java.util.EnumSet;

public class ColGroupLinearFunctionalTest {

	protected static final Log LOG = LogFactory.getLog(ColGroupLinearFunctionalTest.class.getName());

	@Test
	public void testDecompressToDenseBlockSingleColumn() {
//		AColGroup colGroup = ColGroupConst.create(new double[] {3, 4, 5});
//		double[] test = new double[] {0, 0, 0};
//		colGroup.computeColSums(test, 10);
//		System.out.println(colGroup.getNumCols());
//		System.out.println(Arrays.toString(test));

		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}}, true);
	}

	@Test
	public void testDecompressToDenseBlockSingleColumnTransposed() {
		testDecompressToDenseBlock(new double[][] {{1}, {2}, {3}, {4}, {5}}, false);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumns() {
		testDecompressToDenseBlock(new double[][] {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}, false);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumnsUnequalSlopeTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}, {-1, -2, -3, -4, -5}}, true);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumnsTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5}, {1, 1, 1, 1, 1}}, true);
	}

	@Test
	public void testDecompressToDenseBlockTwoColumnsNonLinearTransposed() {
		testDecompressToDenseBlock(new double[][] {{1, 2, 3, 4, 5, 0}, {1, 1, 1, 1, 1, 2}}, true);
	}

	public void testDecompressToDenseBlock(double[][] data, boolean isTransposed) {
		MatrixBlock mbt = DataConverter.convertToMatrixBlock(data);

		final int numCols = isTransposed ? mbt.getNumRows() : mbt.getNumColumns();
		final int numRows = isTransposed ? mbt.getNumColumns() : mbt.getNumRows();
		int[] colIndexes = new int[numCols];
		for(int x = 0; x < numCols; x++)
			colIndexes[x] = x;

		try {
			CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(AColGroup.CompressionType.LinearFunctional)).create();
			cs.transposed = isTransposed;

			final CompressedSizeInfoColGroup cgi = new CompressedSizeEstimatorExact(mbt, cs)
				.getColGroupInfo(colIndexes);
			CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			AColGroup cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);

			// Decompress to dense block
			MatrixBlock ret = new MatrixBlock(numRows, numCols, false);
			ret.allocateDenseBlock();
			cg.decompressToDenseBlock(ret.getDenseBlock(), 0, numRows);

			MatrixBlock expected = DataConverter.convertToMatrixBlock(data);
			if(isTransposed)
				LibMatrixReorg.transposeInPlace(expected, 1);
			Assert.assertArrayEquals(expected.getDenseBlockValues(), ret.getDenseBlockValues(), 0.01);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName());
		}
	}

}
