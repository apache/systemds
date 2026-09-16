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

package org.apache.sysds.test.functions.io.csv;

import java.util.Collections;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameCSVHeaderNamesTest extends AutomatedTestBase {
	private static final String TEST_NAME = "FrameCSVHeaderNames";
	private static final String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR =
		TEST_DIR + FrameCSVHeaderNamesTest.class.getSimpleName() + "/";

	private static final int ROWS = 42;
	private static final String[] COLUMN_NAMES = new String[] {"customer_id", "signup_date", "score"};

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testFrameCSVHeaderNamesCP() {
		runCSVHeaderNamesTest(ExecType.CP);
	}

	@Test
	public void testFrameCSVHeaderNamesSpark() {
		runCSVHeaderNamesTest(ExecType.SPARK);
	}

	private void runCSVHeaderNamesTest(ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		setOutputBuffering(true);
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {
				"-args", input("A"), String.valueOf(ROWS), String.valueOf(COLUMN_NAMES.length), output("B")
			};

			Types.ValueType[] schema = Collections.nCopies(
				COLUMN_NAMES.length, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
			FrameBlock inputFrame = new FrameBlock(schema);
			inputFrame.setColumnNames(COLUMN_NAMES);

			double[][] data = getRandomMatrix(ROWS, schema.length, -10, 10, 0.8, 7);
			TestUtils.initFrameData(inputFrame, data, schema, ROWS);

			FrameWriter writer = FrameWriterFactory.createFrameWriter(
				FileFormat.CSV, new FileFormatPropertiesCSV(true, ",", false));
			writer.writeFrameToHDFS(inputFrame, input("A"), ROWS, schema.length);

			runTest(true, false, null, -1);

			FrameReader reader = FrameReaderFactory.createFrameReader(
				FileFormat.CSV, new FileFormatPropertiesCSV(true, ",", false));
			FrameBlock outputFrame = reader.readFrameFromHDFS(output("B"), ROWS, schema.length);
			Assert.assertArrayEquals("Column names were not preserved across CSV read/write.",
				COLUMN_NAMES, outputFrame.getColumnNames());
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
