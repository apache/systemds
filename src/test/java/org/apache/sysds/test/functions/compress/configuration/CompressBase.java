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

package org.apache.sysds.test.functions.compress.configuration;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.DMLCompressionStatistics;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;

import java.io.ByteArrayOutputStream;

public abstract class CompressBase extends AutomatedTestBase {
	// private static final Log LOG = LogFactory.getLog(CompressBase.class.getName());

	protected abstract String getTestClassDir();

	protected abstract String getTestName();

	protected abstract String getTestDir();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
	}

	public void runTest(int rows, int cols, int decompressCount, int compressCount, ExecType ex, String name) {
		compressTest(rows, cols, 1.0, ex, 1, 5, 1.4, decompressCount, compressCount, name);
	}

	public void compressTest(int rows, int cols, double sparsity, ExecType instType, int min, int max, double delta,
		int decompressionCountExpected, int compressionCountsExpected, String name) {

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;
		Types.ExecMode platformOld = setExecMode(instType);
		try {

			loadTestConfiguration(getTestConfiguration(getTestName()));

			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 42, delta);
			writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, 1000, rows * cols));

			fullDMLScriptName = SCRIPT_DIR + "/functions/compress/compress_" + name + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs", "A=" + input("A")};

			ByteArrayOutputStream tmp = runTest(null);
			String out = tmp != null ? runTest(null).toString() : "";

			int decompressCount = DMLCompressionStatistics.getDecompressionCount();
			long compressionCount = (instType == ExecType.SPARK) ? Statistics
				.getCPHeavyHitterCount("sp_compress") : Statistics.getCPHeavyHitterCount("compress");
			DMLCompressionStatistics.reset();

			Assert.assertEquals(out + "\ncompression count   wrong : ", compressionCount, compressionCountsExpected);
			Assert.assertTrue(out + "\nDecompression count wrong : " + decompressCount +
							(decompressionCountExpected >= 0 ? " [expected: " + decompressionCountExpected+ "]" : ""),
				decompressionCountExpected >= 0 ? decompressionCountExpected == decompressCount : decompressCount > 1);

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
