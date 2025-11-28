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

import java.util.EnumSet;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
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

}
