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

import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
public abstract class JolEstimateTest {

	protected static final CompressionType ddc = CompressionType.DDC;
	protected static final CompressionType ole = CompressionType.OLE;
	protected static final CompressionType rle = CompressionType.RLE;
	protected static final CompressionType unc = CompressionType.UNCOMPRESSED;

	public static long kbTolerance = 1024;

	private static final int seed = 7;
	private final long tolerance;
	private final MatrixBlock mbt;
	private final CompressionSettings cs;
	private final CompressionSettings csl;// Compression Settings Lossy;
	private ColGroup cg;
	private ColGroup cgl; // ColGroup Lossy;

	public abstract CompressionType getCT();

	public JolEstimateTest(MatrixBlock mb, int tolerance) {
		this.mbt = mb;
		this.tolerance = tolerance;
		EnumSet<CompressionType> vc = EnumSet.of(getCT());
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setSeed(seed).setSamplingRatio(1.0)
			.setValidCompressions(vc);
		this.cs = csb.create();
		this.csl = csb.setLossy(true).setSortValuesByLength(false).create();

		int[] colIndexes = new int[mbt.getNumRows()];
		for(int x = 0; x < mbt.getNumRows(); x++) {
			colIndexes[x] = x;
		}
		try {
			ABitmap ubm = BitmapEncoder.extractBitmap(colIndexes, mbt, cs);
			cg = ColGroupFactory.compress(colIndexes, mbt.getNumColumns(), ubm, getCT(), cs, mbt);
			ABitmap ubml = BitmapEncoder.extractBitmap(colIndexes, mbt, csl);
			cgl = ColGroupFactory.compress(colIndexes, mbt.getNumColumns(), ubml, getCT(), csl, mbt);

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed to compress colGroup! " + e.getMessage(), false);
		}
	}

	@Test
	public void compressedSizeInfoEstimatorExact() {
		try {
			// CompressionSettings cs = new CompressionSettings(1.0);
			CompressedSizeEstimator cse = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, cs);
			CompressedSizeInfoColGroup csi = cse.estimateCompressedColGroupSize();
			long estimateCSI = csi.getCompressionSize(getCT());
			long estimateObject = cg.estimateInMemorySize();
			String errorMessage = "CSI estimate " + estimateCSI + " should be exactly " + estimateObject + "\n"
				+ cg.toString();
			boolean res = Math.abs(estimateCSI - estimateObject) <= tolerance;
			if(res && !(estimateCSI == estimateObject)) {
				// Make a warning in case that it is not exactly the same.
				// even if the test allows some tolerance.
				System.out.println("NOT EXACTLY THE SAME! " + this.getClass().getName() + " " + errorMessage);
			}
			assertTrue(errorMessage, res);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed Test", false);
		}
	}

	@Test
	public void compressedSizeInfoEstimatorExactLossy() {
		try {
			// CompressionSettings cs = new CompressionSettings(1.0);
			CompressedSizeEstimator cse = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, csl);
			CompressedSizeInfoColGroup csi = cse.estimateCompressedColGroupSize();
			long estimateCSI = csi.getCompressionSize(getCT());
			long estimateObject = cgl.estimateInMemorySize();
			String errorMessage = "CSI estimate " + estimateCSI + " should be exactly " + estimateObject + "\n"
				+ cg.toString();
			boolean res = Math.abs(estimateCSI - estimateObject) <= tolerance;
			if(res && !(estimateCSI == estimateObject)) {
				// Make a warning in case that it is not exactly the same.
				// even if the test allows some tolerance.
				System.out.println("NOT EXACTLY THE SAME! " + this.getClass().getName() + " " + errorMessage);
			}
			assertTrue(errorMessage, res);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed Test", false);
		}
	}

	// @Test
	// public void compressedSizeInfoEstimatorSampler() {
	// try {
	// CompressionSettings cs = new CompressionSettingsBuilder().copySettings(this.cs).setSamplingRatio(0.1).create();
	// CompressedSizeEstimator cse = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, cs);
	// CompressedSizeInfoColGroup csi = cse.computeCompressedSizeInfos(1).compressionInfo[0];
	// long estimateCSI = csi.getCompressionSize(getCT());
	// long estimateObject = cg.estimateInMemorySize();
	// String errorMessage = "CSI Sampled estimate " + estimateCSI + " should be larger than actual "
	// + estimateObject + " but not more than " + (tolerance + kbTolerance) + " off";
	// if(!(estimateCSI == estimateObject)) {
	// System.out.println("NOT EXACTLY THE SAME IN SAMPLING! " + errorMessage);
	// }
	// boolean res = Math.abs(estimateCSI - estimateObject) <= tolerance + kbTolerance;
	// assertTrue(errorMessage, res);
	// }
	// catch(Exception e) {
	// e.printStackTrace();
	// assertTrue("Failed Test", false);
	// }
	// }
}