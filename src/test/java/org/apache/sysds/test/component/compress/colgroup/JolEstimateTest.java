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

import java.util.EnumSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.lib.BitmapEncoder;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
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

	private final int[] colIndexes;
	private final MatrixBlock mbt;

	public abstract CompressionType getCT();

	private final long actualSize;
	// The actual compressed column group
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
			ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, mbt, true);
			cg = ColGroupFactory.compress(colIndexes, mbt.getNumColumns(), ubm, getCT(), cs, mbt);
			actualSize = cg.estimateInMemorySize();
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new DMLRuntimeException("Failed construction : " + this.getClass().getSimpleName());
		}
	}

	@Test
	public void compressedSizeInfoEstimatorExact() {
		try {
			CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(1.0)
				.setValidCompressions(EnumSet.of(getCT())).setSeed(seed).create();
			cs.transposed = true;

			final long estimateCSI = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, cs)
				.estimateCompressedColGroupSize().getCompressionSize(cg.getCompType());

			boolean res = Math.abs(estimateCSI - actualSize) <= 0;
			assertTrue("CSI estimate " + estimateCSI + " should be exactly " + actualSize + "\n" + cg.toString(), res);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed exact test " + getCT(), false);
		}
	}

	@Test
	public void compressedSizeInfoEstimatorSample_90() {
		compressedSizeInfoEstimatorSample(0.9, 0.9);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_50() {
		compressedSizeInfoEstimatorSample(0.5, 0.90);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_20() {
		compressedSizeInfoEstimatorSample(0.2, 0.8);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_10() {
		compressedSizeInfoEstimatorSample(0.1, 0.75);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_5() {
		compressedSizeInfoEstimatorSample(0.05, 0.7);
	}

	@Test
	public void compressedSizeInfoEstimatorSample_1() {
		compressedSizeInfoEstimatorSample(0.01, 0.6);
	}

	public void compressedSizeInfoEstimatorSample(double ratio, double tolerance) {
		try {
			if(mbt.getNumColumns() < CompressedSizeEstimatorFactory.minimumSampleSize)
				return; // Skip the tests that anyway wouldn't use the sample based approach.

			CompressionSettings cs = new CompressionSettingsBuilder().setSamplingRatio(ratio)
				.setValidCompressions(EnumSet.of(getCT())).setSeed(seed).create();
			cs.transposed = true;

			final CompressedSizeInfoColGroup cgsi = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, cs)
				.estimateCompressedColGroupSize();
			final long estimateCSI = cgsi.getCompressionSize(cg.getCompType());
			final double minTolerance = actualSize * tolerance;
			final double maxTolerance = actualSize / tolerance;
			final String rangeString = minTolerance + " < " + estimateCSI + " < " + maxTolerance;
			boolean res = minTolerance < estimateCSI && estimateCSI < maxTolerance;
			assertTrue(
				"CSI Sampled estimate is not in tolerance range " + rangeString + "\n" + cgsi + "\n" + cg.toString(),
				res);

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed sample test " + getCT() + "", false);
		}
	}

	// Currently ignore because lossy compression is disabled.
	// @Test
	// @Ignore
	// public void compressedSizeInfoEstimatorExactLossy() {
	// try {
	// // CompressionSettings cs = new CompressionSettings(1.0);
	// csl.transposed = true;
	// CompressedSizeEstimator cse = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, csl);
	// CompressedSizeInfoColGroup csi = cse.estimateCompressedColGroupSize();
	// long estimateCSI = csi.getCompressionSize(getCT());
	// long estimateObject = cgl.estimateInMemorySize();

	// String errorMessage = "CSI estimate " + estimateCSI + " should be exactly " + estimateObject + "\n"
	// + cg.toString();
	// boolean res = Math.abs(estimateCSI - estimateObject) <= 0;
	// if(res && !(estimateCSI == estimateObject)) {
	// // Make a warning in case that it is not exactly the same.
	// // even if the test allows some tolerance.
	// System.out.println("NOT EXACTLY THE SAME! " + this.getClass().getName() + " " + errorMessage);
	// }
	// assertTrue(errorMessage, res);
	// }
	// catch(Exception e) {
	// e.printStackTrace();
	// assertTrue("Failed Test", false);
	// }
	// }

}
