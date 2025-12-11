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

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.EnumSet;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSizes;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.ComEstExact;
import org.apache.sysds.runtime.compress.estim.ComEstFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public abstract class JolEstimateTest {

	protected static final Log LOG = LogFactory.getLog(JolEstimateTest.class.getName());

	protected static final CompressionType ddc = CompressionType.DDC;
	protected static final CompressionType delta = CompressionType.DeltaDDC;
	protected static final CompressionType ole = CompressionType.OLE;
	protected static final CompressionType rle = CompressionType.RLE;
	protected static final CompressionType sdc = CompressionType.SDC;
	protected static final CompressionType unc = CompressionType.UNCOMPRESSED;
	private static final int seed = 7;

	private static final CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumSampleSize(100)
		.setSeed(seed);

	private final IColIndex colIndexes;
	private final MatrixBlock mbt;

	public abstract CompressionType getCT();

	protected boolean shouldTranspose() {
		return true;
	}

	private final long actualSize;
	private final int actualNumberUnique;
	private final AColGroup cg;

	public JolEstimateTest(MatrixBlock mbt) {
		CompressedMatrixBlock.debug = true;
		this.mbt = mbt;
		colIndexes = ColIndexFactory.create(shouldTranspose() ? mbt.getNumRows() : mbt.getNumColumns());

		mbt.recomputeNonZeros();
		mbt.examSparsity();
		try {
			CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(getCT()));
			boolean useDelta = getCT() == CompressionType.DeltaDDC;
			if(useDelta)
				csb.setPreferDeltaEncoding(true);
			CompressionSettings cs = csb.create();
			cs.transposed = shouldTranspose();

			final ComEstExact est = new ComEstExact(mbt, cs);
			final CompressedSizeInfoColGroup cgi = useDelta ? est.getDeltaColGroupInfo(colIndexes) : est.getColGroupInfo(colIndexes);

			final CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			final List<AColGroup> groups = ColGroupFactory.compressColGroups(mbt, csi, cs, 1);

			if(groups.size() == 1) {
				cg = groups.get(0);
				actualSize = cg.estimateInMemorySize();
				actualNumberUnique = cg.getNumValues();
			}
			else {
				cg = null;
				actualSize = groups.stream().mapToLong(x -> x.estimateInMemorySize()).sum();
				actualNumberUnique = groups.stream().mapToInt(x -> x.getNumValues()).sum();
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName());
		}
	}

	@Test
	public void compressedSizeInfoEstimatorExact() {
		compressedSizeInfoEstimatorSample(1.0, 1.0);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_90() {
		compressedSizeInfoEstimatorSample(0.9, 0.9);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_50() {
		compressedSizeInfoEstimatorSample(0.5, 0.8);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_20() {
		compressedSizeInfoEstimatorSample(0.2, 0.6);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_10() {
		compressedSizeInfoEstimatorSample(0.1, 0.5);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_5() {
		compressedSizeInfoEstimatorSample(0.05, 0.5);
	}

	// @Test
	// public void compressedSizeInfoEstimatorSample_1() {
	// compressedSizeInfoEstimatorSample(0.01, 0.4);
	// }

	@Test
	public void testToString() {
		// just to add a tests to verify that the to String method does not crash
		if(cg != null)
			cg.toString();
	}

	public void compressedSizeInfoEstimatorSample(double ratio, double tolerance) {
		if(cg == null)
			return;
		try {
			if(mbt.getNumColumns() > 10000)
				tolerance *= 0.95;

			CompressionSettingsBuilder testCsb = csb.setSamplingRatio(ratio).setMinimumSampleSize(10)
				.setValidCompressions(EnumSet.of(getCT()));
			boolean useDelta = getCT() == CompressionType.DeltaDDC;
			if(useDelta)
				testCsb.setPreferDeltaEncoding(true);
			final CompressionSettings cs = testCsb.create();
			cs.transposed = shouldTranspose();

			final int sampleSize = Math.max(10, (int) (mbt.getNumColumns() * ratio));
			final AComEst est = ComEstFactory.createEstimator(mbt, cs, sampleSize, 1);
			final CompressedSizeInfoColGroup cInfo = useDelta ? est.getDeltaColGroupInfo(colIndexes) : est.getColGroupInfo(colIndexes);
			final int estimateNUniques = cInfo.getNumVals();

			final double estimateCSI = (cg.getCompType() == CompressionType.CONST) ? ColGroupSizes
				.estimateInMemorySizeCONST(cg.getNumCols(), true, 1.0,
					false) : cInfo.getCompressionSize(cg.getCompType());
			final double minTolerance = actualSize * tolerance *
				(ratio < 1 && mbt.getSparsity() < 0.8 ? mbt.getSparsity() + 0.2 : 1);
			double maxTolerance = actualSize / tolerance;
			if(cg.getNumValues() > sampleSize / 2)
				maxTolerance += Math.abs(cg.getNumValues() - estimateNUniques) * 8 * mbt.getNumRows();

			if(cg.getCompType() == CompressionType.SDC)
				maxTolerance += 8 * mbt.getNumRows();

			if(cg.getCompType() == CompressionType.RLE)
				maxTolerance += 8 * (mbt.getNumColumns() / Character.MAX_VALUE) * cg.getNumValues();

			final boolean withinToleranceOnSize = minTolerance <= estimateCSI && estimateCSI <= maxTolerance;

			if(!withinToleranceOnSize) {
				final String rangeString = String.format("%.0f <= %.0f <= %.0f , Actual Size %d", minTolerance,
					estimateCSI, maxTolerance, actualSize);
				String message = "CSI Sampled estimate size is not in tolerance range \n" + rangeString
					+ "\nActual number uniques:" + actualNumberUnique + " estimated Uniques: " + estimateNUniques
					+ "\nSampleSize of total rows:: " + sampleSize + " " + mbt.getNumColumns() + "\n" + cInfo + "\n";
				if(mbt.getNumColumns() < 1000)
					message += mbt;

				fail(message + "\n" + cg);
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed sample test " + getCT() + "", false);
		}
	}

	protected static Object[] conv(double[][] vals) {
		return new Object[] {DataConverter.convertToMatrixBlock(vals)};
	}

	protected static Object[] gen(int row, int col, int min, int max, double spar, int seed) {
		return new Object[] {TestUtils.generateTestMatrixBlock(row, col, min, max, spar, seed)};
	}

	protected static Object[] genR(int row, int col, int min, int max, double spar, int seed) {
		return new Object[] {genRM(row, col, min, max, spar, seed)};
	}

	protected static MatrixBlock genRM(int row, int col, int min, int max, double spar, int seed) {
		return TestUtils.ceil(TestUtils.generateTestMatrixBlock(row, col, min, max, spar, seed));
	}
}
