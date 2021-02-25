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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;


public class BuiltinEMATest extends AutomatedTestBase {

	private final static String TEST_NAME = "exponentialMovingAverage";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinOutlierTest.class.getSimpleName() + "/";

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void CompareToAirGap() {
		Double[] disguised_values = new Double[]{.0, .1, .2, .3};
		Double[][] values = new Double[][]{disguised_values};
		FrameBlock f = generateRandomFrameBlock(4, 1, values);
		runMissingValueTest(f, ExecType.CP,  1, "triple", 0);
	}

	private void runMissingValueTest(FrameBlock test_frame, ExecType et, Integer search_iterations, String mode, Integer freq)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "F=" + input("F"), "O=" + output("O"), "search_iterations=" + search_iterations, "mode=" + mode, "freq=" + freq};

			FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV).
					writeFrameToHDFS(test_frame, input("F"), test_frame.getNumRows(), test_frame.getNumColumns());

			runTest(true, false, null, -1);

			FrameBlock outputFrame = readDMLFrameFromHDFS("O", Types.FileFormat.CSV);
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static FrameBlock generateRandomFrameBlock(int rows, int cols, Double[][] values)
	{
		Types.ValueType[] schema = new Types.ValueType[cols];
		for(int i = 0; i < cols; i++) {
			schema[i] = Types.ValueType.FP64;
		}

		if(values != null) {
			String[] names = new String[cols];
			for(int i = 0; i < cols; i++)
				names[i] = schema[i].toString();
			FrameBlock frameBlock = new FrameBlock(schema, names);
			frameBlock.ensureAllocatedColumns(rows);
			for(int row = 0; row < rows; row++)
				for(int col = 0; col < cols; col++)
					frameBlock.set(row, col, values[col][row]);
			return frameBlock;
		}
		return TestUtils.generateRandomFrameBlock(rows, cols, schema ,TestUtils.getPositiveRandomInt());
	}
}
