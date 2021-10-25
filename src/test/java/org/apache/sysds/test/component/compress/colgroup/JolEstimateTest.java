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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.bitmap.BitmapEncoder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorExact;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorSample;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.EstimationFactors;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public abstract class JolEstimateTest {

	protected static final Log LOG = LogFactory.getLog(JolEstimateTest.class.getName());

	protected static final CompressionType ddc = CompressionType.DDC;
	protected static final CompressionType ole = CompressionType.OLE;
	protected static final CompressionType rle = CompressionType.RLE;
	protected static final CompressionType sdc = CompressionType.SDC;
	protected static final CompressionType unc = CompressionType.UNCOMPRESSED;
	private static final int seed = 7;

	private static final CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumSampleSize(100)
		.setSeed(seed);

	private final int[] colIndexes;
	private final MatrixBlock mbt;

	public abstract CompressionType getCT();

	private final long actualSize;
	private final int actualNumberUnique;
	private final AColGroup cg;

	public JolEstimateTest(MatrixBlock mbt) {
		this.mbt = mbt;
		colIndexes = new int[mbt.getNumRows()];
		for(int x = 0; x < mbt.getNumRows(); x++)
			colIndexes[x] = x;

		try {
			CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(getCT())).create();
			cs.transposed = true;
			ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, mbt, true, 8, false);

			EstimationFactors ef = CompressedSizeEstimator.estimateCompressedColGroupSize(ubm, colIndexes,
				mbt.getNumColumns(), cs);
			CompressedSizeInfoColGroup cgi = new CompressedSizeInfoColGroup(colIndexes, ef, getCT());
			CompressedSizeInfo csi = new CompressedSizeInfo(cgi);
			cg = ColGroupFactory.compressColGroups(mbt, csi, cs, 1).get(0);
			// cg = ColGroupFactory.compress(colIndexes, mbt.getNumColumns(), ubm, getCT(), cs, mbt, 1);
			actualSize = cg.estimateInMemorySize();
			actualNumberUnique = cg.getNumValues();
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
	@Ignore
	public void compressedSizeInfoEstimatorSample_90() {
		compressedSizeInfoEstimatorSample(0.9, 0.9);
	}

	@Test
	@Ignore
	public void compressedSizeInfoEstimatorSample_50() {
		compressedSizeInfoEstimatorSample(0.5, 0.8);
	}

	@Test
	@Ignore
	public void compressedSizeInfoEstimatorSample_20() {
		compressedSizeInfoEstimatorSample(0.2, 0.7);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_10() {
		compressedSizeInfoEstimatorSample(0.1, 0.6);
	}

	@Test
	@Ignore
	public void compressedSizeInfoEstimatorSample_5() {
		compressedSizeInfoEstimatorSample(0.05, 0.5);
	}

	@Test
	@Ignore
	public void compressedSizeInfoEstimatorSample_1() {
		compressedSizeInfoEstimatorSample(0.01, 0.4);
	}

	@Test 
	public void testToString(){
		// just to add a tests to verify that the to String method does not crash
		cg.toString();
	}

	public void compressedSizeInfoEstimatorSample(double ratio, double tolerance) {
		try {

			CompressionSettings cs = csb.setSamplingRatio(ratio).setValidCompressions(EnumSet.of(getCT())).create();
			cs.transposed = true;

			CompressedSizeEstimator est = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, cs, 1);
			final int sampleSize = (est instanceof CompressedSizeEstimatorSample) ? ((CompressedSizeEstimatorSample) est)
				.getSampleSize() : est.getNumRows();

			if(est instanceof CompressedSizeEstimatorExact)
				return;
			final CompressedSizeInfoColGroup cgsi = est.estimateCompressedColGroupSize();

			if(cg.getCompType() != CompressionType.UNCOMPRESSED && actualNumberUnique > 10) {

				final int estimateNUniques = cgsi.getNumVals();
				final double minToleranceNUniques = actualNumberUnique * tolerance;
				final double maxToleranceNUniques = actualNumberUnique / tolerance;
				final boolean withinToleranceOnNUniques = minToleranceNUniques <= estimateNUniques &&
					estimateNUniques <= maxToleranceNUniques;

				if(!withinToleranceOnNUniques) {
					final String uniqueString = String.format("%.0f <= %d <= %.0f, Actual %d", minToleranceNUniques,
						estimateNUniques, maxToleranceNUniques, actualNumberUnique);
					fail("CSI Sampled estimate of number of unique values not in range\n" + uniqueString);
				}
			}

			final long estimateCSI = cgsi.getCompressionSize(cg.getCompType());
			final double minTolerance = actualSize * tolerance;
			final double maxTolerance = actualSize / tolerance;
			final boolean withinToleranceOnSize = minTolerance <= estimateCSI && estimateCSI <= maxTolerance;
			if(!withinToleranceOnSize) {
				final String rangeString = String.format("%.0f <= %d <= %.0f , Actual Size %d", minTolerance, estimateCSI,
					maxTolerance, actualSize);

				fail("CSI Sampled estimate size is not in tolerance range \n" + rangeString + "\nActual number uniques:"
					+ actualNumberUnique + "\nSampleSize of total rows:: " + sampleSize + " " + mbt.getNumColumns() + "\n"
					+ cg);
			}

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed sample test " + getCT() + "", false);
		}
	}
}
