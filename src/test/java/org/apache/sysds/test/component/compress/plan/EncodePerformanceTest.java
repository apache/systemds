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

package org.apache.sysds.test.component.compress.plan;

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.plan.CompressionPlanFactory;
import org.apache.sysds.runtime.compress.plan.IPlanEncode;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;

public class EncodePerformanceTest {
	protected static final Log LOG = LogFactory.getLog(EncodePerformanceTest.class.getName());

	public static void main(String[] args) {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(1000, 300, 1, 1, 0.5, 235);
		IPlanEncode plan = CompressionPlanFactory.nCols(mb.getNumColumns(),10, CompressionType.DDC, 16);
		// testExpand(mb, plan);
		testEncode(mb, plan);
	}

	private static void testExpand(MatrixBlock mb, IPlanEncode plan) {
		try {

			for(int j = 0; j < 5; j++) {
				Timing time = new Timing(true);
				plan.expandPlan(mb);
				LOG.error(time.stop());
			}
			for(int i = 0; i < 10000; i++) {
				Timing time = new Timing(true);
				for(int j = 0; j < 100; j++) {
					plan.expandPlan(mb);
				}
				LOG.error(time.stop());
			}
			MatrixBlock cmb = plan.encode(mb);
			TestUtils.compareMatrices(mb, cmb, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	private static void testEncode(MatrixBlock mb, IPlanEncode plan) {
		try {
			MatrixBlock cmb = null;
			plan.expandPlan(mb);
			for(int i = 0; i < 10000; i++) {
				Timing time = new Timing(true);
				for(int j = 0; j < 100; j++) {
					cmb = plan.encode(mb);
				}
				LOG.error(time.stop());
			}
			TestUtils.compareMatrices(mb, cmb, 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}
}
