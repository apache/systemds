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

package org.apache.sysds.test.functions.compress.matrixByBin;

import java.io.IOException;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.builtin.part1.BuiltinDistTest;
import org.junit.Test;


public class CompressByBinTest extends AutomatedTestBase {


	private final static String TEST_NAME = "compressByBins";
	private final static String TEST_DIR = "functions/compress/matrixByBin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDistTest.class.getSimpleName() + "/";

	private final static int rows = 1000;

	private final static int cols = 10;

	private final static int[] dVector = new int[cols];

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"X"}));
	}

	@Test
	public void testCompressBinsDefaultMatrixCP() { runCompress(Types.ExecType.CP); }

	@Test
	public void testCompressBinsDefaultFrameCP() { runCompressFrame(Types.ExecType.CP); }

	private void runCompress(Types.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X")};

			//generate actual dataset
			double[][] X = getRandomMatrix(rows, cols, -100, 100, 1, 7);
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);

		}
		finally {
			rtplatform = platformOld;
		}
	}

	private void runCompressFrame(Types.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("X")};

			//generate actual dataset
			double[][] X = getRandomMatrix(rows, cols, -100, 100, 1, 7);
			writeInputMatrixWithMTD("X", X, true);

			Types.ValueType[] schema = new Types.ValueType[]{Types.ValueType.INT32};
			FrameBlock Xf = TestUtils
				.generateRandomFrameBlock(1000, schema, 7);
			writeInputFrameWithMTD("X", Xf, false, schema, Types.FileFormat.CSV);
			runTest(true, false, null, -1);

		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}