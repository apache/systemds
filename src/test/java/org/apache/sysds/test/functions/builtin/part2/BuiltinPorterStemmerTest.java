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

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.IOException;

public class BuiltinPorterStemmerTest extends AutomatedTestBase {

	private final static String TEST_NAME = "porterStemmerTest";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = BuiltinPorterStemmerTest.class.getSimpleName() + "/";
	private static final String INPUT = DATASET_DIR +"stemming/dictionary.csv";

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
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"D"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}
	@Test
	public void testStemmerCP() {
		runStemmerTest(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testStemmerSpark() {
		runStemmerTest(Types.ExecMode.SPARK);
	}

	private void runStemmerTest(Types.ExecMode et)
	{
		Types.ExecMode modeOld = setExecMode(et);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", INPUT, output("S"), output("E")};

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			FrameBlock outputFrame = readDMLFrameFromHDFS("S", Types.FileFormat.CSV);
			FrameBlock inputFrame = readDMLFrameFromHDFS("E", Types.FileFormat.CSV);
			String[] output = (String[])outputFrame.getColumnData(0);
			String[] input = (String[])inputFrame.getColumnData(0);
			//expected vs stemmer output
			int count = 0;
			for(int i = 0; i<input.length; i++) {
				if(input[i].equals(output[i]))
					count++;
			}
			Assert.assertEquals(110, count, 10);
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
