/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements. See the NOTICE file
* distributed with this work for additional information regarding copyright ownership.
* The ASF licenses this file to you under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software distributed under the License
* is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
* either express or implied. See the License for the specific language governing permissions and limitations
* under the License.
*/

package org.apache.sysds.test.component.compress.qcompress;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupOLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupRLE;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDCSingle;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUncompressed;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.cost.CostEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.ComEstFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class QuantizationFusedForcedCompressionTypesTest {

	private static final int K = 4;
	private static final long SEED = 1234;

	/**
	 * Test 1: Test the Uncompressed column group by directly calling the create method.
	 * 
	 * m0 is generated as a floored matrix. m1 is generated as a full-precision matrix, but will be internally multiplied
	 * by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). The best compression types for both matrices are
	 * DDC, but we explicitly create UNCOMPRESSED columns.
	 * 
	 */
	@Test
	public void testForcedUncompressed() {
		try {
			MatrixBlock m0 = generateTestMatrix(10000, 500, -100, 100, 1.0, SEED, true);
			MatrixBlock m1 = generateTestMatrix(10000, 500, -100, 100, 1.0, SEED, false);

			CompressionSettings cs0 = createCompressionSettings(null);
			CompressionSettings cs1 = createCompressionSettings(new double[] {1.0});

			Pair<CompressedSizeInfo, AComEst> compressedGroupsResult0 = generateCompressedGroups(m0, cs0);
			CompressedSizeInfo compressedGroups0 = compressedGroupsResult0.getLeft();

			Pair<CompressedSizeInfo, AComEst> compressedGroupsResult1 = generateCompressedGroups(m1, cs1);
			CompressedSizeInfo compressedGroups1 = compressedGroupsResult1.getLeft();

			assertEquals("Mismatch in number of compressed groups", compressedGroups0.getInfo().size(),
				compressedGroups1.getInfo().size(), 0.0);

			for(int i = 0; i < compressedGroups0.getInfo().size(); i++) {
				AColGroup colGroup0 = ColGroupUncompressed.create(compressedGroups0.getInfo().get(i).getColumns(), m0,
					cs0.transposed);
				AColGroup colGroup1 = ColGroupUncompressed.createQuantized(compressedGroups1.getInfo().get(i).getColumns(),
					m1, cs1.transposed, cs1.scaleFactors);

				assertEquals("Mismatch in column group sum", colGroup0.getSum(m0.getNumRows()),
					colGroup1.getSum(m1.getNumRows()), 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Compression extraction failed: " + e.getMessage());
		}
	}

	/**
	 * Test 2: Test the RLE compression type by forcing RLE in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored column matrix. m1 is generated as a full-precision column matrix, but will be
	 * internally multiplied by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches
	 * extractBitmapSingleColumn().
	 */
	@Test
	public void testForcedRLETypeSingleColumn() {
		testForcedCompressionTypeSingleColumn(CompressionType.RLE, ColGroupRLE.class);
	}

	/**
	 * Test 3: Test the RLE compression type by forcing RLE in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored matrix. m1 is generated as a full-precision matrix, but will be internally multiplied
	 * by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches extractBitmapMultiColumns().
	 * 
	 */
	@Test
	public void testForcedRLETypeMultiColumn() {
		testForcedCompressionTypeMultiColumn(CompressionType.RLE, ColGroupRLE.class);
	}

	/**
	 * Test 4: Test the OLE compression type by forcing OLE in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored column matrix. m1 is generated as a full-precision column matrix, but will be
	 * internally multiplied by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches
	 * extractBitmapSingleColumn().
	 */
	@Test
	public void testForcedOLETypeSingleColumn() {
		testForcedCompressionTypeSingleColumn(CompressionType.OLE, ColGroupOLE.class);
	}

	/**
	 * Test 5: Test the OLE compression type by forcing OLE in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored matrix. m1 is generated as a full-precision matrix, but will be internally multiplied
	 * by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches extractBitmapMultiColumn().
	 */
	@Test
	public void testForcedOLETypeMultiColumn() {
		testForcedCompressionTypeMultiColumn(CompressionType.OLE, ColGroupOLE.class);
	}

	/**
	 * Test 6: Test the SDC compression type by forcing SDC in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored column matrix. m1 is generated as a full-precision column matrix, but will be
	 * internally multiplied by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches
	 * extractBitmapSingleColumn(). This should also cover CONST, EMPTY, SDCFOR.
	 */
	@Test
	public void testForcedSDCTypeSingleColumn() {
		testForcedCompressionTypeSingleColumn(CompressionType.SDC, ColGroupSDC.class);
	}

	/**
	 * Test 7: Test the SDC compression type by forcing SDC in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored matrix. m1 is generated as a full-precision matrix, but will be internally multiplied
	 * by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches extractBitmapMultiColumn(). This should also
	 * cover CONST, EMPTY, SDCFOR.
	 */
	@Test
	public void testForcedSDCTypeMultiColumn() {
		testForcedCompressionTypeMultiColumn(CompressionType.SDC, ColGroupSDCSingle.class);
	}

	/**
	 * Test 8: Test the DDC compression type by forcing DDC in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored column matrix. m1 is generated as a full-precision column matrix, but will be
	 * internally multiplied by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches
	 * directCompressDDCSingleCol(). This should also cover DDCFOR.
	 */
	@Test
	public void testForcedDDCTypeSingleColumn() {
		testForcedCompressionTypeSingleColumn(CompressionType.DDC, ColGroupDDC.class);
	}

	/**
	 * Test 9: Test the DDC compression type by forcing DDC in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored matrix. m1 is generated as a full-precision matrix, but will be internally multiplied
	 * by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches directCompressDDCMultiCol(). This should
	 * also cover DDCFOR.
	 */
	@Test
	public void testForcedDDCTypeMultiColumn() {
		testForcedCompressionTypeMultiColumn(CompressionType.DDC, ColGroupDDC.class);
	}

	/**
	 * Test the given compression type by forcing it in each CompressedSizeInfoColGroup.
	 * 
	 * m0 is generated as a floored column matrix. m1 is generated as a full-precision column matrix, but will be
	 * internally multiplied by 1.0 and floored. Essentially m0 = floor(m1 * scaleFactor). Reaches
	 * extractBitmapSingleColumn().
	 */
	private void testForcedCompressionTypeSingleColumn(CompressionType compressionType,
		Class<? extends AColGroup> expectedGroupClass) {
		try {
			int nRow = 100;
			int nCol = 1;
			int max = 50;
			int min = -50;
			double s = 1.0;

			MatrixBlock m0 = generateTestMatrix(nRow, nCol, min, max, s, SEED, true);
			MatrixBlock m1 = generateTestMatrix(nRow, nCol, min, max, s, SEED, false);

			CompressionSettings cs0 = createCompressionSettings(null);
			CompressionSettings cs1 = createCompressionSettings(new double[] {1.0});

			List<AColGroup> results0 = compressWithForcedTypeNoCoCode(m0, cs0, compressionType);
			List<AColGroup> results1 = compressWithForcedTypeNoCoCode(m1, cs1, compressionType);

			assertEquals("Mismatch in number of resulting column groups", results0.size(), results1.size(), 0.0);

			for(int i = 0; i < results0.size(); i++) {
				assertInstanceOf(expectedGroupClass, results0.get(i), "Mismatch in forced compression type");
				assertInstanceOf(expectedGroupClass, results1.get(i), "Mismatch in forced compression type");

				assertEquals("Mismatch in sum of values in column group", results0.get(i).getSum(nRow),
					results1.get(i).getSum(nRow), 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Compression extraction failed: " + e.getMessage());
		}
	}

	private void testForcedCompressionTypeMultiColumn(CompressionType compressionType,
		Class<? extends AColGroup> expectedGroupClass) {
		try {
			double[][] values = {{1.5, 2.5, 3.5, 4.5, 5.5}, {1.5, 2.5, 3.5, 4.5, 5.5}, {1.5, 2.5, 3.5, 4.5, 5.5},
				{2.5, 3.5, 4.5, 5.5, 6.5}, {2.5, 3.5, 4.5, 5.5, 6.5}, {2.5, 3.5, 4.5, 5.5, 6.5},};

			int nRow = values.length;

			MatrixBlock m0 = DataConverter.convertToMatrixBlock(values);
			m0 = TestUtils.floor(m0);
			m0.recomputeNonZeros();

			MatrixBlock m1 = DataConverter.convertToMatrixBlock(values);

			CompressionSettings cs0 = createCompressionSettings(null);
			CompressionSettings cs1 = createCompressionSettings(new double[] {1.0});

			List<AColGroup> results0 = compressWithForcedTypeCoCode(m0, cs0, compressionType);
			List<AColGroup> results1 = compressWithForcedTypeCoCode(m1, cs1, compressionType);

			assertEquals("Mismatch in number of resulting column groups", results0.size(), results1.size(), 0.0);

			for(int i = 0; i < results0.size(); i++) {
				assertInstanceOf(expectedGroupClass, results0.get(i), "Mismatch in forced compression type");
				assertInstanceOf(expectedGroupClass, results1.get(i), "Mismatch in forced compression type");
				assertEquals("Mismatch in sum of values in column group", results0.get(i).getSum(nRow),
					results1.get(i).getSum(nRow), 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Compression extraction failed: " + e.getMessage());
		}
	}

	private static void assertInstanceOf(Class<?> expected, Object obj, String message) {
		if(!expected.isInstance(obj)) {
			fail(message + ": Expected " + expected.getSimpleName() + ", but got " + obj.getClass().getSimpleName());
		}
	}

	/**
	 * Generate compressed groups with an estimator.
	 */
	private static Pair<CompressedSizeInfo, AComEst> generateCompressedGroups(MatrixBlock matrix,
		CompressionSettings cs) {
		AComEst estimator = ComEstFactory.createEstimator(matrix, cs, K);
		CompressedSizeInfo sizeInfo = estimator.computeCompressedSizeInfos(K);
		return Pair.of(sizeInfo, estimator);
	}

	/**
	 * Force a specific compression type (e.g., RLE) on a set of compressed groups.
	 */
	private static List<AColGroup> compressWithForcedTypeNoCoCode(MatrixBlock matrix, CompressionSettings cs,
		CompressionType type) {
		Pair<CompressedSizeInfo, AComEst> result = generateCompressedGroups(matrix, cs);
		CompressedSizeInfo originalGroups = result.getLeft();
		List<CompressedSizeInfoColGroup> modifiedGroups = forceCompressionType(originalGroups, type);
		CompressedSizeInfo compressedGroupsNew = new CompressedSizeInfo(modifiedGroups);
		return ColGroupFactory.compressColGroups(matrix, compressedGroupsNew, cs, K);
	}

	/**
	 * Force a specific compression type (e.g., RLE) on a set of compressed groups with CoCode.
	 */
	private static List<AColGroup> compressWithForcedTypeCoCode(MatrixBlock matrix, CompressionSettings cs,
		CompressionType type) {
		Pair<CompressedSizeInfo, AComEst> result = generateCompressedGroups(matrix, cs);
		CompressedSizeInfo originalGroups = result.getLeft();
		AComEst estimator = result.getRight();
		ACostEstimate ice = CostEstimatorFactory.create(cs, null, matrix.getNumRows(), matrix.getNumColumns(),
			matrix.getSparsity());
		originalGroups = CoCoderFactory.findCoCodesByPartitioning(estimator, originalGroups, K, ice, cs);
		List<CompressedSizeInfoColGroup> modifiedGroups = forceCompressionType(originalGroups, type);
		CompressedSizeInfo compressedGroupsNew = new CompressedSizeInfo(modifiedGroups);
		return ColGroupFactory.compressColGroups(matrix, compressedGroupsNew, cs, K);
	}

	/**
	 * Modify the compression type of each group to a specific type.
	 */
	private static List<CompressedSizeInfoColGroup> forceCompressionType(CompressedSizeInfo originalGroups,
		CompressionType type) {
		List<CompressedSizeInfoColGroup> modifiedGroups = new ArrayList<>();
		for(CompressedSizeInfoColGroup cg : originalGroups.getInfo()) {
			Set<CompressionType> compressionTypes = new HashSet<>();
			compressionTypes.add(type);
			modifiedGroups
				.add(new CompressedSizeInfoColGroup(cg.getColumns(), cg.getFacts(), compressionTypes, cg.getMap()));
		}
		return modifiedGroups;
	}

	/**
	 * Generate a test matrix with specified dimensions, value range, and sparsity.
	 */
	private static MatrixBlock generateTestMatrix(int nRow, int nCol, int min, int max, double s, long seed,
		boolean floored) {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(nRow, nCol, min, max, s, seed);
		return floored ? TestUtils.floor(mb) : mb;
	}

	/**
	 * Create compression settings with an optional scale factor.
	 */
	private static CompressionSettings createCompressionSettings(double[] scaleFactor) {
		CompressionSettingsBuilder builder = new CompressionSettingsBuilder();
		// .setColumnPartitioner(PartitionerType.GREEDY).setSeed((int) SEED);
		if(scaleFactor != null) {
			builder.setScaleFactor(scaleFactor);
		}
		return builder.create();
	}
}
