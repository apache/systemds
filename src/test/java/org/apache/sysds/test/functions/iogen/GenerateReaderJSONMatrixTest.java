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

package org.apache.sysds.test.functions.iogen;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.runtime.iogen.ReaderMappingJSON;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class GenerateReaderJSONMatrixTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "functions/iogen/";
	protected final static String TEST_CLASS_DIR = TEST_DIR + GenerateReaderJSONMatrixTest.class.getSimpleName() + "/";
	private final static String TEST_NAME = "GenerateReaderJSONMatrixTest";

	protected String sampleRaw;
	protected double[][] sampleMatrix;

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] {"Y"}));
	}

	@Test public void test1() {
		sampleRaw = "{\"col1\":1, \"col2\":2, \"col34\":{\"col3\":3,\"col4\":4}}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test2() {
		sampleRaw = "{\"col1\":1, \"col2\":2, \"col3\":[3,4,5,6,7], \"col4\":[8,9]}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test3() {
		sampleRaw = "{\n" + "\t\"value1\":{\"a\":1,\"value2\":[[{\"a\":1,\"b\":2},{\"a\":3,\"b\":4,\"c\":5}],[{\"a\":6,\"b\":7},{\"a\":8,\"b\":9,\"c\":10}]]}\n" + "}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test4() {
		sampleRaw = "{\"D1\":1, \"D2\":\"saeed\", \"D3\":{\"a\":2,\"b\":\"b\"}, \"D4\":[{\"f\":6, \"g\":\"g\"}]}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test5() {
		sampleRaw = "{\"D1\":1, \"D2\":\"saeed\", \"D3\":{\"a\":2,\"b\":\"b\"}, \"D4\":[{\"f\":6, \"g\":\"g\"}], \"D5\":[{\"h\":[7,8,9]}]}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test6() {
		sampleRaw = "{\"D1\":1, \"D2\":\"saeed\", \"D3\":{\"a\":2,\"b\":\"b\"}, \"D4\":[{\"f\":6, \"g\":\"g\"}], \"D5\":[{\"h\":[7,8,9], \"k\":{\"l\":10}}]}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test7() {
		sampleRaw = "{\n" + "\t\"fname\": \"saeed\",\n" + "\t\"lname\": \"fathollahzadeh\",\n" + "\t\"birth date\":\"1985-09-28\",\n" + "\t\"addresses\":\n" + "\t{\n" + "\t\t\"street\": \"inf 13\", \"floor\":5\n" + "\t}\n" + "}\n" + "{\n" + "\t\"fname\": \"saeed\",\t\n" + "\t\"list1\":[1,2,3,4,5,6,7],\n" + "\t\"list2\":[{\"a\":1,\"b\":2},{\"a\":3,\"b\":4},{\"a\":5,\"b\":6},{\"a\":7,\"b\":8}],\n" + "\t\"list3\":[[1,2,3],[4,5,6],[7,8,9]],\n" + "\t\"list4\":[[{\"a\":1,\"b\":2},{\"a\":3,\"b\":4,\"c\":5}],[{\"a\":6,\"b\":7},{\"a\":8,\"b\":9,\"c\":10}]],\n" + "\t\"value5\":{\"a\":1,\"vaaaaaa\":[[{\"a\":1,\"b\":2},{\"a\":3,\"b\":4,\"c\":5}],[{\"a\":6,\"b\":7},{\"a\":8,\"b\":9,\"c\":10}]]}\n" + "}\n" + "{\n" + "\t\"fname\": \"saeed\",\n" + "\t\"lname\": \"fathollahzadeh\",\n" + "\t\"birth date\":\"1985-09-28\",\n" + "\t\n" + "}";
		sampleMatrix = new double[][] {{1, 2, 3, 4}};
		runGenerateReaderTest();
	}

	@Test public void test8() {
		sampleRaw = "{\"D1\":1, \"D2\":\"2\", \"D3\":{\"a\":4,\"b\":\"5\"}, \"D4\":[{\"f\":6, \"g\":\"7\"}]}";
		sampleMatrix = new double[][] {{1, 2, 4, 5, 6, 7}};
		runGenerateReaderTest();
	}
	@Test public void test9() {
		sampleRaw = "{\"D1\":1, \"D2\":\"2\", \"D3\":{\"a\":4,\"b\":\"5\"}, \"D4\":[{\"f\":6, \"g\":\"7\"}]}\n" + "{\"D1\":10, \"D3\":{\"a\":40}, \"D4\":[{\"f\":60, \"g\":\"70\"}]}\n" + "{\"D1\":11, \"D2\":\"21\", \"D3\":{\"a\":41,\"b\":\"51\"}}";
		sampleMatrix = new double[][] {{1,2,4,5,6,7},{10,0,40,0,60,70},{11,21,41,51,0,0}};
		runGenerateReaderTest();
	}

	@SuppressWarnings("unused") protected void runGenerateReaderTest() {

		Types.ExecMode oldPlatform = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = false;

			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			MatrixBlock sampleMB = DataConverter.convertToMatrixBlock(sampleMatrix);

			ReaderMappingJSON.MatrixReaderMapping mappingJSON = new ReaderMappingJSON.MatrixReaderMapping(sampleRaw,
				sampleMB);
		}
		catch(Exception exception) {
			exception.printStackTrace();
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
