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
package org.apache.sysds.test.functions.dnn;

import java.util.HashMap;
import java.util.stream.IntStream;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class Conv1DTest extends AutomatedTestBase
{
	private final static String TEST_NAME_1 = "Conv1DTest";
	private final static String TEST_NAME_2 = "Conv1DBackwardDataTest";
	private final static String TEST_NAME_3 = "Conv1DBackwardFilterTest";
	private final static String TEST_DIR = "functions/tensor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + Conv1DTest.class.getSimpleName() + "/";
	private final static double epsilon=0.0000000001;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_1, new String[] {"output"}));
		addTestConfiguration(TEST_NAME_2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_2, new String[] {"output"}));
		addTestConfiguration(TEST_NAME_3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_3, new String[] {"output"}));
	}

	@Test
	public void testSimpleConv1DDenseSingleBatchSingleChannelSingleFilter(){
		int numImg = 1; int imgSize = 4; int numChannels = 1; int numFilters = 1; int filterSize = 2; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		expected.put(new CellIndex(1,1), 3.0);
		expected.put(new CellIndex(1,2), 5.0);
		expected.put(new CellIndex(1,3), 7.0);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense1() {
		int numImg = 5; int imgSize = 3; int numChannels = 3; int numFilters = 3; int filterSize = 2; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpected(expected, 1, 6, 21.0, 39.0);
		fillExpected(expected, 2, 6, 75.0, 93.0);
		fillExpected(expected, 3, 6, 129.0, 147.0);
		fillExpected(expected, 4, 6, 183.0, 201.0);
		fillExpected(expected, 5, 6, 237.0, 255.0);

		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense2() {
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeated(expected, 3, new double[]{136.,264.,392.,520.},1);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense3() {
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 2; int pad = 1;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeated(expected, 3, new double[]{78.,200.,328.,456.,414.},1);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense4() {
		int numImg = 3; int imgSize = 10; int numChannels = 1; int numFilters = 3; int filterSize = 2; int stride = 2; int pad = 1;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeated(expected,3,new double[]{1.,5.,9.,13.,17.,10.},1);
		fillExpectedRepeated(expected,3,new double[]{11.,25.,29.,33.,37.,20.},2);
		fillExpectedRepeated(expected,3,new double[]{21.,45.,49.,53.,57.,30.},3);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense5() {
		int numImg = 3; int imgSize = 8; int numChannels = 2; int numFilters = 3; int filterSize = 3; int stride = 1; int pad = 2;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeated(expected,3,new double[]{3.,10.,21.,33.,45.,57.,69.,81.,58.,31.},1);
		fillExpectedRepeated(expected,3,new double[]{35.,74.,117.,129.,141.,153.,165.,177.,122.,63.},2);
		fillExpectedRepeated(expected,3,new double[]{67.,138.,213.,225.,237.,249.,261.,273.,186.,95.},3);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense6() {
		int numImg = 1; int imgSize = 10; int numChannels = 4; int numFilters = 3; int filterSize = 4; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeated(expected,3,new double[]{136.,200.,264.,328.,392.,456.,520.},1);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense7() {
		int numImg = 3; int imgSize = 64; int numChannels = 1; int numFilters = 3; int filterSize = 2; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		double[] firstExpected = IntStream.iterate(3,n -> n+2).limit(63).mapToDouble(i->(double)i).toArray();
		double[] secondExpected = IntStream.iterate(131,n -> n+2).limit(63).mapToDouble(i->(double)i).toArray();
		double[] thirdExpected = IntStream.iterate(259,n -> n+2).limit(63).mapToDouble(i->(double)i).toArray();
		fillExpectedRepeated(expected,3,firstExpected,1);
		fillExpectedRepeated(expected,3,secondExpected,2);
		fillExpectedRepeated(expected,3,thirdExpected,3);
		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DDense1SP()
	{
		int numImg = 5; int imgSize = 3; int numChannels = 3; int numFilters = 6; int filterSize = 2; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpected(expected, 1, 12, 21.0, 39.0);
		fillExpected(expected, 2, 12, 75.0, 93.0);
		fillExpected(expected, 3, 12, 129.0, 147.0);
		fillExpected(expected, 4, 12, 183.0, 201.0);
		fillExpected(expected, 5, 12, 237.0, 255.0);
		runConv1DTest(ExecType.SPARK, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_1);
	}

	@Test
	public void testConv1DBackwardDataDense1() {
		int numImg = 5; int imgSize = 3; int numChannels = 3; int numFilters = 3; int filterSize = 1; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeated(expected,3,new double[]{6.,15.,24.},1);
		fillExpectedRepeated(expected,3,new double[]{33.,42.,51.},2);
		fillExpectedRepeated(expected,3,new double[]{60.,69.,78.},3);
		fillExpectedRepeated(expected,3,new double[]{87.,96.,105.},4);
		fillExpectedRepeated(expected,3,new double[]{114.,123.,132.},5);

		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_2);
	}

	@Test
	public void testConv1DBackwardFilterDense1() {
		int numImg = 2; int imgSize = 3; int numChannels = 2; int numFilters = 3; int filterSize = 1; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeatedCol(expected,3,new double[]{608.,686.});

		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_3);
	}

	@Test
	public void testConv1DBackwardFilterDense2() {
		int numImg = 2; int imgSize = 3; int numChannels = 2; int numFilters = 3; int filterSize = 2; int stride = 1; int pad = 0;
		HashMap<CellIndex, Double> expected = new HashMap<>();
		fillExpectedRepeatedCol(expected,3,new double[]{680.,888.,784.,992.});

		runConv1DTest(ExecType.CP, imgSize, numImg, numChannels, numFilters, filterSize, stride, pad, expected, TEST_NAME_3);
	}

	private static void fillExpected(HashMap<CellIndex, Double> expected,
		int rowNum, int rowLength, double value1, double value2)
	{
		for ( int m = 1; m <= rowLength; m+=2){
			expected.put(new CellIndex(rowNum,m), value1);
			expected.put(new CellIndex(rowNum,m+1), value2);
		}
	}

	private static void fillExpectedRepeated(HashMap<CellIndex, Double> expected,
		int repetitionNum, double[] values, int row)
	{
		int colPointer = 1;
		for (int i = 1; i <= repetitionNum;i++){
			for(double value : values) {
				expected.put(new CellIndex(row, colPointer), value);
				colPointer++;
			}
		}
	}

	private static void fillExpectedRepeatedCol(HashMap<CellIndex, Double> expected,int repetitionRows, double[] values){
		for ( int i = 1; i <= repetitionRows; i++){
			for ( int j = 1; j <= values.length; j++ ){
				expected.put(new CellIndex(i,j), values[j-1]);
			}
		}
	}

	public void runConv1DTest( ExecType et, int imgSize, int numImg, int numChannels, int numFilters,
		int filterSize, int stride, int pad, HashMap<CellIndex, Double> expected, String TEST_NAME)
	{
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			getAndLoadTestConfiguration(TEST_NAME);

			String SCRIPT_HOME = SCRIPT_DIR + TEST_DIR + TEST_NAME;
			fullDMLScriptName = SCRIPT_HOME + ".dml";

			programArgs = new String[] {
				"-nvargs",
				"imgSize=" + imgSize,
				"numImg=" + numImg,
				"numChannels=" + numChannels,
				"numFilters=" + numFilters,
				"filterSize=" + filterSize,
				"stride=" + stride,
				"pad=" + pad,
				"output=" + output("output")
			};

			// Run DML
			runTest(true, false, null, -1);

			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("output");
			if ( expected != null)
				TestUtils.compareMatrices(dmlfile, expected, epsilon, "B-DML", "B-R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
