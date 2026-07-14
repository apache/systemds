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

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;


@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FrameColumnNamesTest extends AutomatedTestBase {
	private final static String TEST_NAME = "ColumnNames";
	private final static String TEST_NAME_GET = "GetNames";
	private final static String TEST_NAME_SET = "SetNames";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FrameColumnNamesTest.class.getSimpleName() + "/";

	private final static int _rows = 10000;
	@Parameterized.Parameter()
	public String[] _columnNames;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{new String[] {"A", "B", "C"}}, {new String[] {"1", "2", "3"}},
			{new String[] {"Hello", "hello", "Hello", "hi", "u", "w", "u"}},});
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
		addTestConfiguration(TEST_NAME_GET, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_GET, new String[] {"B"}));
		addTestConfiguration(TEST_NAME_SET, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_SET, new String[] {"B"}));

	}

	@Test
	public void testDetectSchemaDoubleCP() {
		runGetColNamesTest(_columnNames, ExecType.CP);
	}

	@Test
	public void testDetectSchemaDoubleSpark() {
		runGetColNamesTest(_columnNames, ExecType.SPARK);
	}

	@Test
	public void testGetNamesCP() {
		runGetNamesTest(_columnNames,  ExecType.CP);
	}

	@Test
	public void testGetNamesSpark() {
		runGetNamesTest(_columnNames,  ExecType.SPARK);
	}

	@Test
	public void testSetNamesCP() {
		runSetNamesTest(_columnNames,  ExecType.CP);
	}

	@Test
	public void testSetNamesSpark() {
		runSetNamesTest(_columnNames,  ExecType.SPARK);
	}

	private void runGetNamesTest(String[] columnNames, ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		if(et == ExecType.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		setOutputBuffering(true);
		try {
			getAndLoadTestConfiguration(TEST_NAME_GET);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME_GET + ".dml";
			programArgs = new String[] {"-args", input("A"), String.valueOf(_rows),
					Integer.toString(columnNames.length), output("B")};

			Types.ValueType[] schema = Collections.nCopies(
					columnNames.length, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
			FrameBlock frame1 = new FrameBlock(schema);
			frame1.setColumnNames(columnNames);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
					new FileFormatPropertiesCSV(true, ",", false));

			double[][] A = getRandomMatrix(_rows, schema.length, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
			TestUtils.initFrameData(frame1, A, schema, _rows);
			writer.writeFrameToHDFS(frame1, input("A"), _rows, schema.length);

			runTest(true, false, null, -1);

			FrameBlock resultFrame =
					readDMLFrameFromHDFS("B", FileFormat.BINARY);

			Assert.assertEquals(
					"Unexpected number of result rows.",
					1,
					resultFrame.getNumRows());

			Assert.assertEquals(
					"Unexpected number of result columns.",
					columnNames.length,
					resultFrame.getNumColumns());

			// verify output schema
			for(int i = 0; i < schema.length; i++) {
				Assert
						.assertEquals("Wrong result: " + columnNames[i] + ".", columnNames[i], resultFrame.get(0, i).toString());
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private void runSetNamesTest(String[] columnNames, ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		if(et == ExecType.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		setOutputBuffering(true);
		try {
			getAndLoadTestConfiguration(TEST_NAME_SET);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME_SET + ".dml";
			programArgs = new String[] {"-args",input("X"),String.valueOf(_rows),Integer.toString(columnNames.length),
					input("N"),output("B")
			};

			Types.ValueType[] schema = Collections.nCopies(
					columnNames.length, Types.ValueType.FP64).toArray(new Types.ValueType[0]);

			FrameBlock frame1 = new FrameBlock(schema);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
					new FileFormatPropertiesCSV(true, ",", false));

			double[][] A = getRandomMatrix(_rows, schema.length, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
			TestUtils.initFrameData(frame1, A, schema, _rows);
			writer.writeFrameToHDFS(frame1, input("X"), _rows, schema.length);

			Types.ValueType[] nameSchema = Collections.nCopies(
					columnNames.length, Types.ValueType.STRING).toArray(new Types.ValueType[0]);

			FrameBlock names = new FrameBlock(nameSchema);
			names.ensureAllocatedColumns(1);
			for(int i = 0; i < columnNames.length; i++)
				names.set(0, i, columnNames[i]);
			FrameWriter nameWriter = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
					new FileFormatPropertiesCSV(false, ",", false));
			nameWriter.writeFrameToHDFS(names, input("N"), 1, columnNames.length);

			runTest(true, false, null, -1);

			FrameBlock frame2 = readDMLFrameFromHDFS("B", FileFormat.BINARY);
			for(int i = 0; i < columnNames.length; i++)
				Assert.assertEquals("Wrong result: " + columnNames[i] + ".", columnNames[i], frame2.get(0, i).toString());
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}


	private void runGetColNamesTest(String[] columnNames, ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		setOutputBuffering(true);
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("A"), String.valueOf(_rows),
				Integer.toString(columnNames.length), output("B")};

			Types.ValueType[] schema = Collections.nCopies(
				columnNames.length, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
			FrameBlock frame1 = new FrameBlock(schema);
			frame1.setColumnNames(columnNames);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.CSV,
				new FileFormatPropertiesCSV(true, ",", false));

			double[][] A = getRandomMatrix(_rows, schema.length, Double.MIN_VALUE, Double.MAX_VALUE, 0.7, 14123);
			TestUtils.initFrameData(frame1, A, schema, _rows);
			writer.writeFrameToHDFS(frame1, input("A"), _rows, schema.length);

			runTest(true, false, null, -1);
			FrameBlock frame2 = readDMLFrameFromHDFS("B", FileFormat.BINARY);

			// verify output schema
			for(int i = 0; i < schema.length; i++) {
				Assert
					.assertEquals("Wrong result: " + columnNames[i] + ".", columnNames[i], frame2.get(0, i).toString());
			}
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
