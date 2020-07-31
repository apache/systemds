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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.BitmapEncoder;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.UncompressedBitmap;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroup.CompressionType;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.openjdk.jol.datamodel.X86_64_DataModel;
import org.openjdk.jol.info.ClassLayout;
import org.openjdk.jol.layouters.HotSpotLayouter;
import org.openjdk.jol.layouters.Layouter;

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
	private final int[] sizes;
	private ColGroup cg;

	public abstract CompressionType getCT();

	public JolEstimateTest(MatrixBlock mb, int[] sizes, int tolerance) {
		this.mbt = mb;
		this.sizes = sizes;
		this.tolerance = tolerance;
		List<CompressionType> vc = new ArrayList<>();
		vc.add(getCT());
		this.cs = new CompressionSettingsBuilder().setSeed(seed).setSamplingRatio(1.0).setValidCompressions(vc).create();

		int[] colIndexes = new int[mbt.getNumRows()];
		for(int x = 0; x < mbt.getNumRows(); x++) {
			colIndexes[x] = x;
		}
		try {
			UncompressedBitmap ubm = BitmapEncoder.extractBitmap(colIndexes, mbt, cs);
			cg = ColGroupFactory.compress(colIndexes, mbt.getNumColumns(), ubm, getCT(), cs, mbt);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed to compress colgroup! " + e.getMessage(), false);
		}
	}

	@Test
	@Ignore //TODO this method is a maintenance obstacle (e.g., why do we expect int arrays in the number of rows?)
	public void instanceSize() {
		assertTrue("Failed Test, because ColGroup is null", cg != null);
		try {
			Layouter l = new HotSpotLayouter(new X86_64_DataModel());
			long jolEstimate = 0;
			long diff;
			StringBuilder sb = new StringBuilder();
			Object[] contains;
			if(cg.getCompType() == ddc) {
				if(sizes[0] < 256) {
					contains = new Object[] {cg, new int[mbt.getNumRows()], new double[sizes[0]],
						new byte[mbt.getNumColumns()]};
				}
				else {
					contains = new Object[] {cg, new int[mbt.getNumRows()], new double[sizes[0]],
						new char[mbt.getNumColumns()]};
				}
			}
			else if(cg.getCompType() == ole) {
				contains = new Object[] {cg, new int[mbt.getNumRows()], new double[sizes[0]], new int[sizes[1]],
					new char[sizes[2]], new int[sizes[3]]};
			}
			else if(cg.getCompType() == rle) {
				contains = new Object[] {cg, new int[mbt.getNumRows()], new double[sizes[0]], new int[sizes[1]],
					new char[sizes[2]]};
			}
			else if(cg.getCompType() == unc) {
				// Unlike the other tests, in the uncompressed col groups it is assumed that the MatrixBlock default
				// implementation estimates correctly.
				// Thereby making this test only fail in cases where the estimation error is located inside the
				// compression package.
				jolEstimate += MatrixBlock.estimateSizeInMemory(mbt.getNumColumns(), mbt.getNumRows(), mbt.getSparsity());
				contains = new Object[] {cg, new int[mbt.getNumRows()]};
			}
			else {
				throw new NotImplementedException("Not Implemented Case for JolEstimate Test");
			}

			for(Object ob : contains) {
				ClassLayout cl = ClassLayout.parseInstance(ob, l);
				diff = cl.instanceSize();
				jolEstimate += diff;
				sb.append(ob.getClass());
				sb.append("  TOTAL MEM: " + jolEstimate + " diff " + diff + "\n");
			}
			long estimate = cg.estimateInMemorySize();
			String errorMessage = " estimate " + estimate + " should be equal to JOL " + jolEstimate + "\n";
			assertTrue(errorMessage + sb.toString() + "\n" + cg.toString(), estimate == jolEstimate);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Failed Test: " + e.getMessage(), false);
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

	// @Test
	// public void compressedSizeInfoEstimatorSampler() {
	// 	try {
	// 		CompressionSettings cs = new CompressionSettingsBuilder().copySettings(this.cs).setSamplingRatio(0.1).create();
	// 		CompressedSizeEstimator cse = CompressedSizeEstimatorFactory.getSizeEstimator(mbt, cs);
	// 		CompressedSizeInfoColGroup csi = cse.computeCompressedSizeInfos(1).compressionInfo[0];
	// 		long estimateCSI = csi.getCompressionSize(getCT());
	// 		long estimateObject = cg.estimateInMemorySize();
	// 		String errorMessage = "CSI Sampled estimate " + estimateCSI + " should be larger than actual "
	// 			+ estimateObject + " but not more than " + (tolerance + kbTolerance) + " off";
	// 		if(!(estimateCSI == estimateObject)) {
	// 			System.out.println("NOT EXACTLY THE SAME IN SAMPLING! " + errorMessage);
	// 		}
	// 		boolean res = Math.abs(estimateCSI - estimateObject) <= tolerance + kbTolerance;
	// 		assertTrue(errorMessage, res);
	// 	}
	// 	catch(Exception e) {
	// 		e.printStackTrace();
	// 		assertTrue("Failed Test", false);
	// 	}
	// }
}