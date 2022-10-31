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

package org.apache.sysds.test.functions.frame;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameDropInvalidLengthTest extends AutomatedTestBase {
	private final static String TEST_NAME = "DropInvalidLength";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FrameDropInvalidLengthTest.class.getSimpleName() + "/";

	private final static int rows = 800;
	private final static int cols = 4;
	private final static Types.ValueType[] schemaStrings = {Types.ValueType.FP64, Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.INT64};

	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testTwoBadColCP() {
		double[][] invalidLength =  {{-1,30,20,-1}};
		runDropInvalidLenTest( invalidLength,1, ExecType.CP);
	}

	@Test
	public void testTwoBadColSP() {
		double[][] invalidLength =  {{-1,30,20,-1}};
		runDropInvalidLenTest( invalidLength,1, ExecType.SPARK);
	}

	@Test
	public void testOneBadColCP() {
		double[][] invalidLength =  {{-1,-1,20,-1}};
		runDropInvalidLenTest( invalidLength,2, ExecType.CP);
	}

	@Test
	public void testOneBadColSP() {
		double[][] invalidLength =  {{-1,-1,20,-1}};
		runDropInvalidLenTest( invalidLength,2, ExecType.SPARK);
	}

	@Test
	public void testAllBadColCP() {
		double[][] invalidLength =  {{2,2,2,1}};
		runDropInvalidLenTest( invalidLength,3, ExecType.CP);
	}

	@Test
	public void testAllBadColSP() {
		double[][] invalidLength =  {{2,2,2,1}};
		runDropInvalidLenTest( invalidLength,3, ExecType.SPARK);
	}

	@Test
	public void testNoneBadColCP() {
		double[][] invalidLength =  {{-1,20,20,-1}};
		runDropInvalidLenTest( invalidLength,4, ExecType.CP);
	}

	@Test
	public void testNoneBadColSP() {
		double[][] invalidLength =  {{-1,20,20,-1}};
		runDropInvalidLenTest( invalidLength,4, ExecType.SPARK);
	}

	private void runDropInvalidLenTest(double[][] colInvalidLength, int test, ExecType et)
	{
		Types.ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("A"), input("M"),
					String.valueOf(rows), Integer.toString(cols), output("B")};
			FrameBlock frame1 = new FrameBlock(schemaStrings);
			double[][] A = getRandomMatrix(rows, cols, 10, 100, 1, 2373);
			initFrameDataString(frame1,A, schemaStrings); // initialize a frame with one column
			FrameWriter writer = FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV);

			List<Integer> badIndex = getBadIndexes(rows/4);
			int expected = 0;

			switch (test) { //Double in String
				case 1:
					for (int i = 0; i < badIndex.size(); i++) {
						frame1.set(badIndex.get(i),1,"This is a very long sentence that could" +
								" count up to multiple characters");
					}
					expected += badIndex.size();
				case 2:
					for (int i = 0; i < badIndex.size(); i++) {
						frame1.set(badIndex.get(i), 2, "This is out of length");
					}
					expected += badIndex.size();
					break;
				case 3:
					expected += rows*cols;
					break;
				case 4:
					expected += 0.0;
					break;
			}
			// write data frame
			writer.writeFrameToHDFS(
					frame1.slice(0, rows - 1, 0, cols-1, new FrameBlock()),
					input("A"), rows, schemaStrings.length);

			// write expected feature length vector
			writeInputMatrixWithMTD("M", colInvalidLength, true);

			runTest(true, false, null, -1);

			// compare output
			FrameBlock frameOut = readDMLFrameFromHDFS("B", Types.FileFormat.BINARY);

			//read output data and compare results
			ArrayList<Object> data = new ArrayList<>();
			for (int i = 0; i < frameOut.getNumRows(); i++)
				for(int j=0; j < frameOut.getNumColumns(); j++ )
					data.add(frameOut.get(i, j));

			int nullNum = Math.toIntExact(data.stream().filter(s -> s == null).count());
			Assert.assertEquals(expected, nullNum, 1e-5);
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}

	private static List<Integer> getBadIndexes(int length) {
		ArrayList<Integer> list = new ArrayList<>();
		for(int i =0; i<length; i++)
			list.add(ThreadLocalRandom.current().nextInt(0, rows));
		return list.stream().distinct().collect(Collectors.toList());
	}

	public static void initFrameDataString(FrameBlock frame1, double[][] data, Types.ValueType[] lschema) {
		for (int j = 0; j < lschema.length; j++) {
			Types.ValueType vt = lschema[j];
			switch (vt) {
				case STRING:
					String[] tmp1 = new String[rows];
					for (int i = 0; i < rows; i++)
						tmp1[i] = (String) UtilFunctions.doubleToObject(vt, data[i][j]);
					frame1.appendColumn(tmp1);
					break;
				case INT64:
					long[] tmp4 = new long[rows];
					for (int i = 0; i < rows; i++)
						data[i][j] = tmp4[i] = (Long) UtilFunctions.doubleToObject(Types.ValueType.INT64,
								data[i][j], false);
					frame1.appendColumn(tmp4);
					break;
				case FP64:
					double[] tmp6 = new double[rows];
					for (int i = 0; i < rows; i++)
						tmp6[i] = (Double) UtilFunctions.doubleToObject(vt, data[i][j], false);
					frame1.appendColumn(tmp6);
					break;
				default:
					throw new RuntimeException("Unsupported value type: " + vt);
			}
		}
	}
}

