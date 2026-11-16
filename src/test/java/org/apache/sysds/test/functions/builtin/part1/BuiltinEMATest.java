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

package org.apache.sysds.test.functions.builtin.part1;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.HashMap;
import java.util.TreeMap;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;


public class BuiltinEMATest extends AutomatedTestBase {

	private final static String TEST_NAME = "exponentialMovingAverage";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinEMATest.class.getSimpleName() + "/";
	private static final double e = 0.00001;

	private Double[] DATA = new Double[]{41.7275, 24.0418, 32.3281, 37.3287, 46.2132, 29.3463, 36.4829, 42.9777, 48.9015,
		31.1802, 37.7179, 40.4202, 51.2069, 31.8872, 40.9783, 43.7725, 55.5586, 33.8509, 42.0764, 45.6423, 59.7668,
		35.1919, 44.3197, 47.9137};

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
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void CompareToAirPython() {
		Double[] data= new Double[]{112.,118.,132.,129.,121.,135.,148.,148.,136.,119.,104.,118.,115.,126.,141.,135.,125.,
			149.,170.,170.,158.,133.,114.,140.,145.,150.,178.,163.,172.,178.,199.,199.,184.,162.,146.,166.,171.,180.,193.,
			181.,183.,218.,230.,242.,209.,191.,172.,194.,196.,196.,236.,235.,229.,243.,264.,272.,237.,211.,180.,201.,204.,
			188.,235.,227.,234.,264.,302.,293.,259.,229.,203.,229.,242.,233.,267.,269.,270.,315.,364.,347.,312.,274.,237.,
			278.,284.,277.,317.,313.,318.,374.,413.,405.,355.,306.,271.,306.,315.,301.,356.,348.,355.,422.,465.,467.,404.,
			347.,305.,336.,340.,318.,362.,348.,363.,435.,491.,505.,404.,359.,310.,337.,360.,342.,406.,396.,420.,472.,548.,
			559.,463.,407.,362.,405.,417.,391.,419.,461.,472.,535.,622.,606.,508.,461.,390.,432.};
		Double[][] values = new Double[][]{data};
		FrameBlock f = generateBlock(data.length, 1, values);
		runTest(f, ExecType.CP,  100, "triple", 12, 0.5, 0.7, 0.1, e);
	}

	@Test
	public void testSimpleExponentialSmoothing() {
		Double[][] values = new Double[][]{DATA};
		FrameBlock f = generateBlock(DATA.length, 1, values);
		runTest(f, ExecType.CP,  100, "single", 4, 0.5, Double.NaN, Double.NaN, e);
	}

	@Test
	public void testDoubleExponentialSmoothing() {
		Double[][] values = new Double[][]{DATA};
		FrameBlock f = generateBlock(DATA.length, 1, values);
		runTest(f, ExecType.CP,  100, "double", 4, 0.5, 0.5, Double.NaN, e);
	}

	@Test
	public void testTripleExponentialSmoothing() {
		Double[][] values = new Double[][]{DATA};
		FrameBlock f = generateBlock(DATA.length, 1, values);
		runTest(f, ExecType.CP,  100, "triple", 4, 0.5, 0.5, 0.5, e);
	}

	@Test
	public void testTripleExponentialSmoothingOddFreq() {
		Double[][] values = new Double[][]{DATA};
		FrameBlock f = generateBlock(DATA.length, 1, values);
		runTest(f, ExecType.CP,  100, "triple", 7, 0.5, 0.5, 0.5, e);
	}

	private static double calcRMSE(Double[] list1, Double[] list2) {
		double sum = 0;
		int minLen = Math.min(list1.length, list2.length);
		for(int counter = 0; counter < minLen; counter++)
			sum += Math.pow(list1[counter] - list2[counter], 2);
		return Math.sqrt(sum / minLen);
	}

	private void runTest(FrameBlock test_frame, ExecType et, Integer search_iterations, String mode, Integer freq,
		double alpha, double beta, double gamma, double max_error)
	{
		Types.ExecMode platformOld = setExecMode(et);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "F=" + input("F"), "O=" + output("O"),
				"search_iterations=" + search_iterations, "mode=" + mode, "freq=" + freq,
				"alpha=" + alpha, "beta=" + beta, "gamma=" + gamma};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir(), mode, freq.toString(), Double.toString(alpha),
				Double.toString(beta), Double.toString(gamma));
			Types.ValueType[] schema = new Types.ValueType[]{Types.ValueType.STRING};
			writeInputFrameWithMTD("F", test_frame, true, schema, Types.FileFormat.CSV);

			runTest(true, false, null, -1);
			runRScript(true);

			FrameBlock outputFrame = readDMLFrameFromHDFS("O", Types.FileFormat.CSV);
			Array<?> values =  outputFrame.getColumn(0);
			Double[] data = new Double[values.size()];
			for(int i = 0; i < values.size(); i++)
				data[i] = Double.valueOf(values.get(i).toString());

			HashMap<CellIndex, Double> rfile = readRMatrixFromExpectedDir("O");
			TreeMap<Integer, Double> treemap = new TreeMap<>();
			for(CellIndex ci : rfile.keySet())
				treemap.put(ci.row, rfile.get(ci));
			Double reference[] = treemap.values().toArray(Double[]::new);

			double rmse = calcRMSE(data, reference);
			assertTrue("RMSE of " + rmse + " exceeds the maximal error of " + max_error, rmse <= max_error);
		}
		catch (Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static FrameBlock generateBlock(int rows, int cols, Double[][] values)
	{
		Types.ValueType[] schema = new Types.ValueType[cols];
		for(int i = 0; i < cols; i++) {
			schema[i] = Types.ValueType.FP64;
		}

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
}
