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

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.DMLCompressionStatistics;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;

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

	public void transpose(int decompressCount, int compressCount) {
		// Currently the transpose would decompress the compression.
		// But since this script only contain one operation on potentially compressed, it should not try to compress but
		// will if forced.
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		compressTest(1, 1000, 1.0, ex, 1, 10, 1, decompressCount, compressCount, "transpose");
	}

	public void sum(int decompressCount, int compressCount) {
		// Only using sum operations the compression should not be decompressed.
		// But since this script only contain one operation on potentially compressed, it should not try to compress but
		// will if forced.
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		compressTest(1, 1000, 1.0, ex, 1, 10, 1, decompressCount, compressCount, "sum");
	}

	public void rowAggregate(int decompressCount, int compressCount) {
		// If we use row aggregates, it is preferable not to compress at all.
		// But since this script only contain one operation on potentially compressed, it should not try to compress but
		// will if forced.
		LopProperties.ExecType ex = LopProperties.ExecType.CP;
		compressTest(1, 1000, 1.0, ex, 1, 10, 1, decompressCount, compressCount, "row_min");
	}

	public void compressTest(int cols, int rows, double sparsity, LopProperties.ExecType instType, int min, int max,
		double delta, int decompressionCountExpected, int compressionCountsExpected, String name) {

		Types.ExecMode platformOld = setExecMode(instType);
		try {

			loadTestConfiguration(getTestConfiguration(getTestName()));

			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 42, delta);
			writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, 1024, rows * cols));

			fullDMLScriptName = SCRIPT_DIR + "/functions/compress/compress_" + name + ".dml";

			programArgs = new String[] {"-stats", "100", "-nvargs", "A=" + input("A")};

			runTest(null);

			int decompressCount = 0;
			decompressCount += DMLCompressionStatistics.getDecompressionCount();
			decompressCount += DMLCompressionStatistics.getDecompressionSTCount();
			long compressionCount = Statistics.getCPHeavyHitterCount("compress");

			Assert.assertEquals(compressionCount, compressionCountsExpected);
			Assert.assertEquals(decompressionCountExpected, decompressCount);

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
