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

package org.apache.sysds.test.component.compress.cost;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory;
import org.apache.sysds.runtime.compress.cocode.CoCoderFactory.PartitionerType;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupFactory;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.ComEstFactory;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Test;

public abstract class ACostTest {

	protected final Log LOG = LogFactory.getLog(ACostTest.class.getName());

	final MatrixBlock mb;
	final ACostEstimate ce;
	final int seed;

	private boolean debug = false;

	public ACostTest(MatrixBlock mb, ACostEstimate costEstimator, int seed) {
		this.mb = mb;
		this.ce = costEstimator;
		this.seed = seed;
	}

	@Test
	public void testCostEstimate() {
		try {
			// Using greedy because it finds good joins and make the test more robust
			CompressionSettings cs = new CompressionSettingsBuilder() // Settings
				.setColumnPartitioner(PartitionerType.GREEDY).setSeed(seed).create();
			int k = 1;
			AComEst ie = ComEstFactory.createEstimator(mb, cs, k);
			final int nRows = mb.getNumRows();
			// Compress individual
			CompressedSizeInfo individualGroups = ie.computeCompressedSizeInfos(k);
			double costUncompressed = ce.getCost(mb);
			double estimatedCostIndividual = ce.getCost(individualGroups);
			List<AColGroup> individualCols = ColGroupFactory.compressColGroups(mb, individualGroups, cs, k);
			double actualCostIndividual = ce.getCost(individualCols, nRows);

			// cocode
			CompressedSizeInfo cocodeGroups = CoCoderFactory.findCoCodesByPartitioning(ie, individualGroups, k, ce, cs);
			double estimatedCostCoCode = ce.getCost(cocodeGroups);
			List<AColGroup> cocodeCols = ColGroupFactory.compressColGroups(mb, cocodeGroups, cs, k);
			double actualCostCoCode = ce.getCost(cocodeCols, nRows);

			if(debug) {
				StringBuilder sb = new StringBuilder();
				sb.append("\nCost Test using:         " + ce);
				sb.append(String.format("\nUncompressedCost:      %15.0f", costUncompressed));
				sb.append(String.format("\nEstimateIndividualCost:%15.0f", estimatedCostIndividual));
				sb.append(String.format("\nActualCostIndividual:  %15.0f", actualCostIndividual));
				sb.append(String.format("\nEstimateCoCodeCost:    %15.0f", estimatedCostCoCode));
				sb.append(String.format("\nActualCoCodeCost:      %15.0f", actualCostCoCode));

				LOG.error(sb);
			}

			// not really sure what to test for and assert, currently this test just verify that there is a cost
			assertTrue("estimated individual cost is negative", estimatedCostIndividual > 0);
			assertTrue("actual individual cost is negative", actualCostIndividual > 0);
			assertTrue("estimated cocode cost is negative", estimatedCostCoCode > 0);
			assertTrue("actual cocode cost is negative", actualCostCoCode > 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed to extract cost");
		}
	}

}
