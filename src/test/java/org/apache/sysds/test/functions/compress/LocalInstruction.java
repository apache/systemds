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

package org.apache.sysds.test.functions.compress;

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;

public abstract class LocalInstruction extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(LocalInstruction.class.getName());

	private final MatrixBlock X;

	public LocalInstruction() {
		X = TestUtils.round(TestUtils.generateTestMatrixBlock(1000, 10, 0, 5, 1.0, 7));
	}

	protected abstract String getTestDir();

	protected void run(String testName, int compressionCount) {
		run(testName, compressionCount, 10);
	}

	protected void run(String testName, int compressionCount, int sparkCollectionCount) {
		ExecMode mode = ExecMode.SPARK;
		ExecMode oldPlatform = setExecMode(mode);

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;
		OptimizerUtils.ALLOW_SCRIPT_LEVEL_LOCAL_COMMAND = true;

		try {
			loadTestConfiguration(getTestConfiguration(testName));

			String HOME = SCRIPT_DIR + getTestDir();
			fullDMLScriptName = HOME + testName + ".dml";
			programArgs = new String[] {"-explain", "-stats", "20", "-args", input("X"), sparkCollectionCount + ""};

			writeInputMatrixWithMTD("X", X, false);

			String ret = runTest(null).toString();
			long actualCompressionCount = Statistics.getCPHeavyHitterCount("sp_compress");
			long actualCollectCount = Statistics.getSparkCollectCount();
			Assert.assertEquals(ret + "Compression count is incorrect", compressionCount, actualCompressionCount);
			Assert.assertEquals(ret + "Collection count is incorrect", sparkCollectionCount, actualCollectCount);
			Statistics.reset();

		}
		catch(Exception e) {
			resetExecMode(oldPlatform);
			e.printStackTrace();
			fail("Failed workload test");
		}
		finally {
			resetExecMode(oldPlatform);
		}
	}
}
