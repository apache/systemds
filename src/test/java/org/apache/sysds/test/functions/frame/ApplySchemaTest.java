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

public class ApplySchemaTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(ApplySchemaTest.class.getName());

	private final static String TEST_NAME = "applySchema";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ApplySchemaTest.class.getSimpleName() + "/";

	private final static int rows = 10;
	// constructed schema
	private final static Types.ValueType[] schemaStrings = {Types.ValueType.INT32, Types.ValueType.BITSET,
		Types.ValueType.FP64};

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

	enum TestType {
		STRING_IN_DOUBLE, DOUBLE_IN_INT, NONE
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S", "F"}));
		if(TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	// @Test
	// public void testApplyWithStrings() {
	// runApplySchemaTest(schemaStrings, rows, schemaStrings.length, TestType.STRING_IN_DOUBLE, 3, ExecType.CP);
	// }

	// @Test
	// public void testApplyWithDoublesInInts() {
	// runApplySchemaTest(schemaStrings, rows, schemaStrings.length, TestType.DOUBLE_IN_INT, 1, ExecType.CP);
	// }

	@Test
	public void testApplyWithNone() {
		runApplySchemaTest(schemaStrings, rows, schemaStrings.length, TestType.NONE, 1, ExecType.CP);
	}

	private void runApplySchemaTest(Types.ValueType[] schema, int rows, int cols, TestType test, int colNum,
		ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		setOutputBuffering(true);
		try {

			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("A"), String.valueOf(rows), Integer.toString(cols),
				String.valueOf(test), String.valueOf(colNum), output("S"), output("F")};
			FrameBlock frame1 = new FrameBlock(schema);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV);

			double[][] A = getRandomMatrix(rows, 3, -Float.MAX_VALUE, Float.MAX_VALUE, 0.7, 2373);
			initFrameDataString(frame1, A, schema, rows, 3);
			writer.writeFrameToHDFS(frame1.slice(0, rows - 1, 0, schema.length - 1, new FrameBlock()), input("A"), rows,
				schema.length);
			runTest(true, false, null, -1);

			FrameBlock detectedSchema = readDMLFrameFromHDFS("S", FileFormat.BINARY);
			FrameBlock changedFrame = readDMLFrameFromHDFS("F", FileFormat.BINARY);

			// verify output schema
			for(int i = 0; i < schema.length; i++) {
				String detected = detectedSchema.get(0, i).toString();
				String applied = changedFrame.getSchema()[i].toString();
				Assert.assertEquals("Wrong result column : " + i + ".", detected, applied);
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
						tmp1[i] = (String) UtilFunctions.doubleToObject(vt, data[i][j]);
					frame1.appendColumn(tmp1);
					break;
				case BITSET:
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
					frame1.appendColumn(tmp4);
					break;
				case FP32:
					double[] tmp5 = new double[rows];
					for(int i = 0; i < rows; i++)
						tmp5[i] = (Float) UtilFunctions.doubleToObject(vt, data[i][j], false);
					frame1.appendColumn(tmp5);
					break;
				case FP64:
					double[] tmp6 = new double[rows];
					for(int i = 0; i < rows; i++)
						tmp6[i] = (Double) UtilFunctions.doubleToObject(vt, data[i][j], false);
					frame1.appendColumn(tmp6);
					break;
				default:
					throw new RuntimeException("Unsupported value type: " + vt);
			}
		}
	}
}
