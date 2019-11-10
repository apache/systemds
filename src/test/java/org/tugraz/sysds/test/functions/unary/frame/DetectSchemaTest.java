/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.unary.frame;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.io.FrameWriter;
import org.tugraz.sysds.runtime.io.FrameWriterFactory;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.security.SecureRandom;

//TODO fix the tests
//TODO move to package frame
public class DetectSchemaTest extends AutomatedTestBase {
	private final static String TEST_NAME = "DetectSchema";
	private final static String TEST_DIR = "functions/unary/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + DetectSchemaTest.class.getSimpleName() + "/";

	private final static int rows = 120;
	private final static Types.ValueType[] schemaStrings = {Types.ValueType.INT32, Types.ValueType.BOOLEAN, Types.ValueType.FP32, Types.ValueType.STRING};
	private final static Types.ValueType[] schemaDoubles = new Types.ValueType[]{Types.ValueType.FP64, Types.ValueType.FP64};
	private final static Types.ValueType[] schemaMixed = new Types.ValueType[]{Types.ValueType.INT64, Types.ValueType.FP64, Types.ValueType.INT64, Types.ValueType.BOOLEAN};

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
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
		if (TEST_CACHE_ENABLED) {
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
	public void testDetectSchemaStringCP() {
		runDetectSchemaTest(schemaStrings, rows, schemaStrings.length, true, ExecType.CP);
	}

	@Test
	public void testDetectSchemaStringSpark() {
		runDetectSchemaTest(schemaStrings, rows, schemaStrings.length, true, ExecType.SPARK);
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
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "recompile_runtime", "-args", input("A"), String.valueOf(rows), Integer.toString(cols), output("B")};
			FrameBlock frame1 = new FrameBlock(schema);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(OutputInfo.CSVOutputInfo);

			if (!isStringTest) {
				double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 2373);
				initFrameDataDouble(frame1, A, schema);
				writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);
			} else {
				double[][] A = getRandomMatrix(rows, schema.length - 1, -10, 10, 0.9, 2373);
				initFrameDataString(frame1, A, schema);
				writer.writeFrameToHDFS(frame1.slice(0, 119, 0, 3, new FrameBlock()), input("A"), rows, schema.length);
			}

			runTest(true, false, null, -1);
			FrameBlock frame2 = readDMLFrameFromHDFS("B", InputInfo.BinaryBlockInputInfo);

			//verify output schema
			for (int i = 0; i < schema.length; i++) {
				Assert.assertEquals("Wrong result: " + frame2.getSchema()[i] + ".",
						schema[i].toString(), frame2.get(0, i).toString());
			}
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

	private void initFrameDataString(FrameBlock frame1, double[][] data, Types.ValueType[] lschema) {
		for (int j = 0; j < lschema.length - 1; j++) {
			Types.ValueType vt = lschema[j];
			switch (vt) {
				case STRING:
					String[] tmp1 = new String[rows];
					for (int i = 0; i < rows; i++)
						tmp1[i] = (String) UtilFunctions.doubleToObject(vt, data[i][j]);
					frame1.appendColumn(tmp1);
					break;
				case BOOLEAN:
					boolean[] tmp2 = new boolean[rows];
					for (int i = 0; i < rows; i++)
						data[i][j] = (tmp2[i] = (Boolean) UtilFunctions.doubleToObject(vt, data[i][j], false)) ? 1 : 0;
					frame1.appendColumn(tmp2);
					break;
				case INT32:
					int[] tmp3 = new int[rows];
					for (int i = 0; i < rows; i++)
						data[i][j] = tmp3[i] = (Integer) UtilFunctions.doubleToObject(Types.ValueType.INT32, data[i][j], false);
					frame1.appendColumn(tmp3);
					break;
				case INT64:
					long[] tmp4 = new long[rows];
					for (int i = 0; i < rows; i++)
						data[i][j] = tmp4[i] = (Long) UtilFunctions.doubleToObject(Types.ValueType.INT64, data[i][j], false);
					frame1.appendColumn(tmp4);
					break;
				case FP32:
					double[] tmp5 = new double[rows];
					for (int i = 0; i < rows; i++)
						tmp5[i] = (Float) UtilFunctions.doubleToObject(vt, data[i][j], false);
					frame1.appendColumn(tmp5);
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
		String[] randomData = generateRandomString(8, rows);
		frame1.appendColumn(randomData);
	}

	private static void initFrameDataDouble(FrameBlock frame, double[][] data, Types.ValueType[] lschema) {
		Object[] row1 = new Object[lschema.length];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < lschema.length; j++) {
				data[i][j] = UtilFunctions.objectToDouble(lschema[j],
					row1[j] = UtilFunctions.doubleToObject(lschema[j], data[i][j]));
			}
			frame.appendRow(row1);
		}
	}

	public static String[] generateRandomString(int stringLength, int rows) {
		String CHAR_LOWER = "abcdefghijklmnopqrstuvwxyz";
		String CHAR_UPPER = CHAR_LOWER.toUpperCase();
		String NUMBER = "0123456789";
		String DATA_FOR_RANDOM_STRING = CHAR_LOWER + CHAR_UPPER + NUMBER;
		String[] A = new String[rows];
		SecureRandom random = new SecureRandom();

		if (stringLength < 1) throw new IllegalArgumentException();
		for (int j = 0; j < rows; j++) {
			StringBuilder sb = new StringBuilder(stringLength);
			for (int i = 0; i < stringLength; i++) {
				int rndCharAt = random.nextInt(DATA_FOR_RANDOM_STRING.length());
				char rndChar = DATA_FOR_RANDOM_STRING.charAt(rndCharAt);
				sb.append(rndChar);
			}
			A[j] = sb.toString();
		}
		return A;
	}
}
