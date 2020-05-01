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

package org.apache.sysds.test.componentx.compress;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionStatistics;
import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.test.component.compress.CompressedMatrixTest;
import org.apache.sysds.test.component.compress.TestConstants.MatrixTypology;
import org.apache.sysds.test.component.compress.TestConstants.SparsityType;
import org.apache.sysds.test.component.compress.TestConstants.ValueRange;
import org.apache.sysds.test.component.compress.TestConstants.ValueType;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.openjdk.jol.datamodel.X86_64_DataModel;
import org.openjdk.jol.info.ClassLayout;
import org.openjdk.jol.layouters.HotSpotLayouter;
import org.openjdk.jol.layouters.Layouter;

@RunWith(value = Parameterized.class)
public class CompressedMatrixSizeTest extends CompressedMatrixTest {

	public CompressedMatrixSizeTest(SparsityType sparType, ValueType valType, ValueRange valRange,
		CompressionSettings compSettings, MatrixTypology matrixTypology) {
		super(sparType, valType, valRange, compSettings, matrixTypology);
	}

	@Test
	public void testCompressionEstimationVSJolEstimate() {
		try {
			if(!(cmb instanceof CompressedMatrixBlock))
				return;
			CompressionStatistics cStat = ((CompressedMatrixBlock) cmb).getCompressionStatistics();
			long actualSize = cStat.size;
			long originalSize = cStat.originalSize;
			long JolEstimatedSize = getJolSize(((CompressedMatrixBlock) cmb));

			StringBuilder builder = new StringBuilder();
			builder.append("\n\t" + String.format("%-40s - %12d", "Actual compressed size: ", actualSize));
			builder.append("\n\t" + String.format("%-40s - %12d", "<= Original size: ", originalSize));
			builder.append("\n\t" + String.format("%-40s - %12d", "and equal to JOL Size: ", JolEstimatedSize));
			// builder.append("\n\t " + getJolSizeString(cmb));
			builder.append("\n\tcol groups types: " + cStat.getGroupsTypesString());
			builder.append("\n\tcol groups sizes: " + cStat.getGroupsSizesString());
			builder.append("\n\t" + this.toString());

			assertTrue(builder.toString(), actualSize == JolEstimatedSize && actualSize <= originalSize);
			// assertTrue(builder.toString(), groupsEstimate < actualSize && colsEstimate < groupsEstimate);

		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException(this.toString() + "\n" + e.getMessage(), e);
		}
	}

	private static long getJolSize(CompressedMatrixBlock cmb) {
		Layouter l = new HotSpotLayouter(new X86_64_DataModel());
		long jolEstimate = 0;
		CompressionStatistics cStat = cmb.getCompressionStatistics();
		for(Object ob : new Object[] {cmb, cStat, cStat.getColGroups(), cStat.getTimeArrayList(), cmb.getColGroups()}) {
			jolEstimate += ClassLayout.parseInstance(ob, l).instanceSize();
		}
		for(ColGroup cg : cmb.getColGroups()) {
			jolEstimate += cg.estimateInMemorySize();
		}
		return jolEstimate;
	}

	@SuppressWarnings("unused")
	private static String getJolSizeString(CompressedMatrixBlock cmb) {
		StringBuilder builder = new StringBuilder();
		Layouter l = new HotSpotLayouter(new X86_64_DataModel());
		long diff;
		long jolEstimate = 0;
		CompressionStatistics cStat = cmb.getCompressionStatistics();
		for(Object ob : new Object[] {cmb, cStat, cStat.getColGroups(), cStat.getTimeArrayList(), cmb.getColGroups()}) {
			ClassLayout cl = ClassLayout.parseInstance(ob, l);
			diff = cl.instanceSize();
			jolEstimate += diff;
			builder.append(cl.toPrintable());
			builder.append("TOTAL MEM: " + jolEstimate + " diff " + diff + "\n");
		}
		for(ColGroup cg : cmb.getColGroups()) {
			diff = cg.estimateInMemorySize();
			jolEstimate += diff;
			builder.append(cg.getCompType());
			builder.append("TOTAL MEM: " + jolEstimate + " diff " + diff + "\n");
		}
		return builder.toString();
	}
}
