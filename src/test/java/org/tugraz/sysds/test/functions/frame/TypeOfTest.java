/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
package org.tugraz.sysds.test.functions.frame;

import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.io.*;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class TypeOfTest extends AutomatedTestBase {
	private final static String TEST_NAME = "TypeOf";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TypeOfTest.class.getSimpleName() + "/";

	private final static Types.ValueType[] schemaStrings = new Types.ValueType[]{Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.STRING};
	private final static Types.ValueType[] schemaMixed = new Types.ValueType[]{Types.ValueType.STRING, Types.ValueType.FP64, Types.ValueType.INT64, Types.ValueType.BOOLEAN};

	private final static int rows = 50;

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

	private static void initFrameData(FrameBlock frame, double[][] data, Types.ValueType[] lschema) {
		Object[] row1 = new Object[lschema.length];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < lschema.length; j++)
				data[i][j] = UtilFunctions.objectToDouble(lschema[j],
					row1[j] = UtilFunctions.doubleToObject(lschema[j], data[i][j]));
			frame.appendRow(row1);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B"}));
		if (TEST_CACHE_ENABLED)
			setOutAndExpectedDeletionDisabled(true);
	}

	@Test
	public void testTypeOfCP() {
		runtypeOfTest(schemaStrings, rows, schemaStrings.length, ExecType.CP);
	}

	@Test
	public void testTypeOfSpark() {
		runtypeOfTest(schemaStrings, rows, schemaStrings.length, ExecType.SPARK);
	}

	@Test
	public void testTypeOfCPD2() {
		runtypeOfTest(schemaMixed, rows, schemaMixed.length, ExecType.CP);
	}

	@Test
	public void testTypeOfSparkD2() {
		runtypeOfTest(schemaMixed, rows, schemaMixed.length, ExecType.SPARK);
	}

	private void runtypeOfTest(Types.ValueType[] schema, int rows, int cols, ExecType et) {
		if (et == ExecType.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"), String.valueOf(rows), String.valueOf(cols), output("B")};
			//data generation
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 2373);
			FrameBlock frame1 = new FrameBlock(schema);
			initFrameData(frame1, A, schema);

			//write frame data to hdfs
			FrameWriter writer = FrameWriterFactory.createFrameWriter(OutputInfo.CSVOutputInfo);
			writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);
			//write meta file
			HDFSTool.writeMetaDataFile(input("A.mtd"), Types.ValueType.FP64, schema, Types.DataType.FRAME, new MatrixCharacteristics(rows, schema.length, 1000), OutputInfo.CSVOutputInfo);

			//run testcase
			runTest(true, false, null, -1);

			//read frame data from hdfs (not via readers to test physical schema)
			FrameReader reader = FrameReaderFactory.createFrameReader(InputInfo.BinaryBlockInputInfo);
			FrameBlock frame2 = ((FrameReaderBinaryBlock) reader).readFirstBlock(output("B"));

			//verify output schema
			for (int i = 0; i < schema.length; i++) {
				Assert.assertEquals("Wrong result: " + frame2.getSchema()[i] + ".",
					schema[i].toString(), frame2.get(0, i));
			}
		}
		catch (Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
