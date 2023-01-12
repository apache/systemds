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

import java.security.SecureRandom;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;

public class DetectSchemaTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(DetectSchemaTest.class.getName());
	private final static String TEST_NAME = "DetectSchema";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + DetectSchemaTest.class.getSimpleName() + "/";

	private final static int rows = 100;
	private final static Types.ValueType[] schemaDoubles = new Types.ValueType[] {Types.ValueType.FP64,
		Types.ValueType.FP64};
	private final static Types.ValueType[] schemaMixed = new Types.ValueType[] {Types.ValueType.INT64,
		Types.ValueType.FP64, Types.ValueType.INT64, Types.ValueType.BOOLEAN};

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if(TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
		if(TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testDetectSchemaDoubleCP() {
		runDetectSchemaTest(schemaDoubles, rows, schemaDoubles.length, false, ExecType.CP);
	}

	@Test
	public void testDetectSchemaDoubleSpark() {
		runDetectSchemaTest(schemaDoubles, rows, schemaDoubles.length, false, ExecType.SPARK);
	}

	@Test
	public void testDetectSchemaMixCP() {
		runDetectSchemaTest(schemaMixed, rows, schemaMixed.length, false, ExecType.CP);
	}

	@Test
	public void testDetectSchemaMixSpark() {
		runDetectSchemaTest(schemaMixed, rows, schemaMixed.length, false, ExecType.SPARK);
	}

	private void runDetectSchemaTest(Types.ValueType[] schema, int rows, int cols, boolean isStringTest, ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		setOutputBuffering(true);
		try {

			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "recompile_runtime", "-args", input("A"), String.valueOf(rows),
				Integer.toString(cols), output("B")};
			FrameBlock frame1 = new FrameBlock(schema);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV);

			double[][] A = getRandomMatrix(rows, schema.length, -Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 2373);
			initFrameDataDouble(frame1, A, schema);
			writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);

			runTest(null);
			FrameBlock frame2 = readDMLFrameFromHDFS("B", FileFormat.BINARY);

			// verify output schema
			for(int i = 0; i < schema.length; i++) {
				Assert.assertEquals("Wrong result column " + i + " : " + frame2.getSchema()[i] + ".", schema[i].toString(),
					frame2.get(0, i).toString());
			}
		}
		catch(Exception ex) {
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

	public static void initFrameDataString(FrameBlock frame1, double[][] data, Types.ValueType[] lschema, int rows,
		int cols) {
		for(int j = 0; j < cols; j++) {
			Types.ValueType vt = lschema[j];
			switch(vt) {
				case STRING:
					String[] tmp1 = new String[rows];
					for(int i = 0; i < rows; i++)
						tmp1[i] = (String) UtilFunctions.doubleToObject(vt, data[i][j]) + "AB";
					frame1.appendColumn(tmp1);
					break;
				case BOOLEAN:
					boolean[] tmp2 = new boolean[rows];
					for(int i = 0; i < rows; i++)
						data[i][j] = (tmp2[i] = (Boolean) UtilFunctions.doubleToObject(vt, data[i][j], false)) ? 1 : 0;
					frame1.appendColumn(tmp2);
					break;
				case INT32:
					int[] tmp3 = new int[rows];
					for(int i = 0; i < rows; i++)
						data[i][j] = tmp3[i] = (Integer) UtilFunctions.doubleToObject(Types.ValueType.INT32, data[i][j],
							false);
					frame1.appendColumn(tmp3);
					break;
				case INT64:
					long[] tmp4 = new long[rows];
					for(int i = 0; i < rows; i++)
						data[i][j] = tmp4[i] = (Long) UtilFunctions.doubleToObject(Types.ValueType.INT64, data[i][j], false);
					data[0][j] = tmp4[0] = ((long) Integer.MAX_VALUE) + 6L;
					frame1.appendColumn(tmp4);
					break;
				case FP32:
					float[] tmp5 = new float[rows];
					for(int i = 0; i < rows; i++)
						data[i][j] = tmp5[i] = (Float) UtilFunctions.doubleToObject(vt, data[i][j], false);
					frame1.appendColumn(tmp5);
					break;
				case FP64:
					double[] tmp6 = new double[rows];
					for(int i = 0; i < rows; i++)
						data[i][j] = tmp6[i] = (Double) UtilFunctions.doubleToObject(vt, data[i][j], false);
					frame1.appendColumn(tmp6);
					break;
				default:
					throw new RuntimeException("Unsupported value type: " + vt);
			}
		}
		String[] randomData = generateRandomString(8, rows);
		frame1.appendColumn(randomData);
		frame1.appendColumn(doubleSpecialData(rows));
		frame1.appendColumn(floatLimitData(rows));
	}

	private static void initFrameDataDouble(FrameBlock frame, double[][] data, Types.ValueType[] lschema) {
		Object[] row1 = new Object[lschema.length];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < lschema.length; j++) {
				data[i][j] = UtilFunctions.objectToDouble(lschema[j],
					row1[j] = UtilFunctions.doubleToObject(lschema[j], data[i][j]));
			}
			frame.appendRow(row1);
		}
	}

	private static String[] generateRandomString(int stringLength, int rows) {
		String CHAR_LOWER = "abcdefghijklmnopqrstuvwxyz";
		String CHAR_UPPER = CHAR_LOWER.toUpperCase();
		String NUMBER = "0123456789";
		String DATA_FOR_RANDOM_STRING = CHAR_LOWER + CHAR_UPPER + NUMBER;
		String[] A = new String[rows];
		SecureRandom random = new SecureRandom();

		if(stringLength < 1)
			throw new IllegalArgumentException();
		for(int j = 0; j < rows; j++) {
			StringBuilder sb = new StringBuilder(stringLength);
			for(int i = 0; i < stringLength; i++) {
				int rndCharAt = random.nextInt(DATA_FOR_RANDOM_STRING.length());
				char rndChar = DATA_FOR_RANDOM_STRING.charAt(rndCharAt);
				sb.append(rndChar);
			}
			A[j] = sb.toString();
		}
		return A;
	}

	private static String[] doubleSpecialData(int rows) {
		String[] dataArray = new String[] {"Infinity", "3.4028234e+38", "Nan", "-3.4028236e+38"};
		String[] A = new String[rows];
		SecureRandom random = new SecureRandom();
		for(int j = 0; j < rows; j++)
			A[j] = dataArray[random.nextInt(4)];
		return A;
	}

	private static double[] floatLimitData(int rows) {
		double[] dataArray = new double[] {Float.MAX_VALUE, 3.4028233E38, 3.4028234e38, 3.4028228e38, 2.4028228e38,
			-3.4028234e38, -3.40282310e38};
		double[] A = new double[rows];
		SecureRandom random = new SecureRandom();
		for(int j = 0; j < rows; j++)
			A[j] = dataArray[random.nextInt(7)];
		return A;
	}
}
