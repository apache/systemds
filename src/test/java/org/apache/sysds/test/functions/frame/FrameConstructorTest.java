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

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class FrameConstructorTest extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(FrameConstructorTest.class.getName());

	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "FrameConstructorTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameConstructorTest.class.getSimpleName() + "/";

	private final static int rows = 40;
	private final static int cols = 4;

	private final static ValueType[] schemaStrings1 = new ValueType[]{
		ValueType.INT64, ValueType.STRING, ValueType.FP64, ValueType.BOOLEAN};

	private final static ValueType[] schemaStrings2 = new ValueType[]{
		ValueType.INT64, ValueType.STRING, ValueType.FP64, ValueType.STRING};

	private enum TestType {
		NAMED,
		NO_SCHEMA,
		RANDOM_DATA,
		SINGLE_DATA,
		MULTI_ROW_DATA,
		UNKNOWN_DIMS,
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
	public void testFrameNamedParam() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, rows,"mixed");
		runFrameTest(TestType.NAMED, exp, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testFrameNamedParamSP() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, rows,"mixed");
		runFrameTest(TestType.NAMED, exp, Types.ExecMode.SPARK);
	}

	@Test
	public void testNoSchema() {
		FrameBlock exp = createExpectedFrame(schemaStrings2, rows,"mixed");
		runFrameTest(TestType.NO_SCHEMA, exp, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testNoSchemaSP() {
		FrameBlock exp = createExpectedFrame(schemaStrings2, rows,"mixed");
		runFrameTest(TestType.NO_SCHEMA, exp, Types.ExecMode.SPARK);
	}

	@Test
	public void testRandData() {
		FrameBlock exp = UtilFunctions.generateRandomFrameBlock(rows, cols, schemaStrings1, new Random(10));
		runFrameTest(TestType.RANDOM_DATA, exp, Types.ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testRandDataSP() {
		FrameBlock exp = UtilFunctions.generateRandomFrameBlock(rows, cols, schemaStrings1, new Random(10));
		runFrameTest(TestType.RANDOM_DATA, exp, Types.ExecMode.SPARK);
	}

	@Test
	public void testSingleData() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, rows,"constant");
		runFrameTest(TestType.SINGLE_DATA, exp, Types.ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testSingleDataSP() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, rows,"constant");
		runFrameTest(TestType.SINGLE_DATA, exp, Types.ExecMode.SPARK);
	}

	@Test
	public void testMultiRowData() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, 5,"multi-row");
		runFrameTest(TestType.MULTI_ROW_DATA, exp, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testMultiRowDataSP() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, 5,"multi-row");
		runFrameTest(TestType.MULTI_ROW_DATA, exp, Types.ExecMode.SPARK);
	}
	
	@Test
	public void testUnknownDims() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, rows,"constant");
		runFrameTest(TestType.UNKNOWN_DIMS, exp, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void testUnknownDimsSP() {
		FrameBlock exp = createExpectedFrame(schemaStrings1, rows, "constant");
		runFrameTest(TestType.UNKNOWN_DIMS, exp, Types.ExecMode.SPARK);
	}
	
	private void runFrameTest(TestType type, FrameBlock expectedOutput, Types.ExecMode et) {
		Types.ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		setOutputBuffering(true);
		try {
			//setup testcase
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-args", String.valueOf(type), output("F2")};


			runTest(null);

			FrameBlock fB = FrameReaderFactory
				.createFrameReader(Types.FileFormat.CSV)
				.readFrameFromHDFS(output("F2"), rows, cols);

			if( type == TestType.MULTI_ROW_DATA)
				fB = fB.slice(0, expectedOutput.getNumRows() -1);

			TestUtils.compareFramesAsString(expectedOutput, fB, false);
			int nrow = type == TestType.MULTI_ROW_DATA ? 5 : 40;
			checkDMLMetaDataFile("F2", new MatrixCharacteristics(nrow, cols));
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
	
	private static FrameBlock createExpectedFrame(ValueType[] schema, int rows, String type) {
		FrameBlock exp = new FrameBlock(schema);
		String[] out = null;
		if(type.equals("mixed"))
			out = new String[]{"1", "abc", "2.5", "TRUE"};
		else if(type.equals("constant"))
			out = new String[]{"1", "1", "1", "1"};
		else if (type.equals("multi-row")) //multi-row data
			out = new String[]{"1", "abc", "2.5", "TRUE"};
		else {
			throw new RuntimeException("invalid test type");
		}

		for(int i=0; i<rows; i++)
			exp.appendRow(out);
		return exp;
	}
}
