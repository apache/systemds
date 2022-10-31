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
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
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

import edu.emory.mathcs.backport.java.util.Collections;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FrameColumnNamesTest extends AutomatedTestBase {
	private final static String TEST_NAME = "ColumnNames";
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
	}

	@Test
	public void testDetectSchemaDoubleCP() {
		runGetColNamesTest(_columnNames, ExecType.CP);
	}

	@Test
	public void testDetectSchemaDoubleSpark() {
		runGetColNamesTest(_columnNames, ExecType.SPARK);
	}

	@SuppressWarnings("unchecked")
	private void runGetColNamesTest(String[] columnNames, ExecType et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", input("A"), String.valueOf(_rows),
				Integer.toString(columnNames.length), output("B")};

			Types.ValueType[] schema = (Types.ValueType[]) Collections
				.nCopies(columnNames.length, Types.ValueType.FP64).toArray(new Types.ValueType[0]);
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
