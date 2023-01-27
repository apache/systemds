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
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BuiltinUNetExtrapolateTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_NAME = "BuiltinUNetExtrapolateTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinUNetExtrapolateTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"x_out"}));
	}

	@Test
	public void extrapolateMultiDimMultiChannel(){
		int hin = 100;
		int win = 100;
		int channels = 3;
		int rows = 10;
		runGenericTest(rows, hin, win, channels);
	}

	@Test
	public void extrapolateSingleChannel(){
		int hin = 100;
		int win = 100;
		int channels = 1;
		int rows = 10;
		runGenericTest(rows, hin, win, channels);
	}

	@Test
	public void extrapolateSingleRow(){
		int hin = 100;
		int win = 100;
		int channels = 3;
		int rows = 1;
		runGenericTest(rows, hin, win, channels);
	}

	private void runGenericTest(int rows, int hin, int win, int channels){
		int cols = hin*win*channels;
		double[][] input = getRandomMatrix(rows, cols,1,10,0.9,3);
		int colsExpected = (hin+184)*(win+184)*channels; //padded height x padded width x number of channels

		getAndLoadTestConfiguration(TEST_NAME);
		setExecMode(Types.ExecMode.SINGLE_NODE);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		String inputName = "features";
		String outputName = "x_out";
		String rowsName = "rows";
		String hinName = "hin";
		String winName = "win";
		String channelName = "channels";
		String useParforName = "useParfor";
		ArrayList<String> programArgsBase = new ArrayList<>(Arrays.asList(
			"-nvargs",
			inputName + "=" + input(inputName),
			rowsName + "=" + rows,
			hinName + "=" + hin,
			winName + "=" + win,
			channelName + "=" + channels
		));
		ArrayList<String> programArgsParfor = new ArrayList<>(programArgsBase);
		programArgsParfor.addAll(List.of(outputName + "=" + output(outputName), useParforName + "=" + "TRUE"));
		ArrayList<String> programArgsFor = new ArrayList<>(programArgsBase);
		programArgsFor.addAll(List.of(outputName + "=" + expected(outputName), useParforName + "=" + "FALSE"));
		programArgs = programArgsParfor.toArray(new String[8]);
		writeInputMatrixWithMTD(inputName,input,false);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		MatrixCharacteristics mc = readDMLMetaDataFile(outputName);
		Assert.assertEquals(
			"Number of rows should be equal to expected number of rows",
			rows, mc.getRows());
		Assert.assertEquals(
			"Number of cols should be equal to expected number of cols",
			colsExpected, mc.getCols());

		programArgs = programArgsFor.toArray(new String[8]);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

		compareResults(1e-9,"parfor", "for");
	}
}
