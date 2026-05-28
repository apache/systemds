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

package org.apache.sysds.test.functions.transform;

import static org.junit.Assert.fail;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class GetCategoricalMaskTest extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(GetCategoricalMaskTest.class.getName());

	private final static String TEST_NAME1 = "GetCategoricalMaskTest";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeApplyTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"y"}));
	}

	@Test
	public void testRecode() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(10, new ValueType[] {ValueType.UINT8}, 32);
		MatrixBlock expected = new MatrixBlock(1, 1, 1.0);
		String spec = "{\"ids\": true, \"recode\": [1]}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testRecode2() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(10, new ValueType[] {ValueType.UINT8, ValueType.UINT8}, 32);
		MatrixBlock expected = new MatrixBlock(1, 2, new double[] {0, 1});

		String spec = "{\"ids\": true, \"recode\": [2]}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testDummy1() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(5, new ValueType[] {ValueType.UINT8, ValueType.INT64}, 32);
		MatrixBlock expected = new MatrixBlock(1, 6, new double[] {0, 1, 1, 1, 1, 1});

		String spec = "{\"ids\": true, \"dummycode\": [2]}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testDummy2() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(5, new ValueType[] {ValueType.UINT8, ValueType.INT64}, 32);
		MatrixBlock expected = new MatrixBlock(1, 6, new double[] {1, 1, 1, 1, 1, 0});

		String spec = "{\"ids\": true, \"dummycode\": [1]}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testHash1() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(5, new ValueType[] {ValueType.UINT8, ValueType.INT64}, 32);
		MatrixBlock expected = new MatrixBlock(1, 4, new double[] {1, 1, 1, 0});

		String spec = "{\"ids\": true, \"dummycode\": [1], \"hash\": [1], \"K\": 3}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testHash2() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT8, ValueType.INT64}, 32);
		MatrixBlock expected = new MatrixBlock(1, 4, new double[] {1, 1, 1, 0});

		String spec = "{\"ids\": true, \"dummycode\": [1], \"hash\": [1], \"K\": 3}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testHash3() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT8, ValueType.INT64,ValueType.UINT8}, 32);
		MatrixBlock expected = new MatrixBlock(1, 7, new double[] {1, 1, 1, 0, 1, 1, 1});

		String spec = "{\"ids\": true, \"dummycode\": [1,3], \"hash\": [1,3], \"K\": 3}";
		runTransformTest(fb, spec, expected);

	}


	@Test
	public void testHybrid1() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT8, ValueType.INT64,ValueType.UINT8, ValueType.BOOLEAN}, 32);
		MatrixBlock expected = new MatrixBlock(1, 9, new double[] {1, 1, 1, 0, 1, 1, 1,1,1});

		String spec = "{\"ids\": true, \"dummycode\": [1,3,4], \"hash\": [1,3], \"K\": 3}";
		runTransformTest(fb, spec, expected);

	}

	@Test
	public void testHybrid2() throws Exception {
		FrameBlock fb = TestUtils.generateRandomFrameBlock(100, new ValueType[] {ValueType.UINT8, ValueType.BOOLEAN,ValueType.UINT8, ValueType.BOOLEAN}, 32);
		MatrixBlock expected = new MatrixBlock(1, 10, new double[] {1, 1, 1, 1,1, 1, 1, 1,1,1});

		String spec = "{\"ids\": true, \"dummycode\": [1,2,3,4], \"hash\": [1,3], \"K\": 3}";
		runTransformTest(fb, spec, expected);

	}

	private void runTransformTest(FrameBlock fb, String spec, MatrixBlock expected) throws Exception {
		try {

			getAndLoadTestConfiguration(TEST_NAME1);
			
			String inF = input("F-In");
			String inS = input("spec");

			TestUtils.writeTestFrame(inF, fb, fb.getSchema(), FileFormat.CSV);
			TestUtils.writeTestScalar(input("spec"), spec);

			String out = output("ret");

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-args", inF, inS, out, expected.getNumColumns() + ""};

			runTest(true, false, null, -1);

			MatrixBlock result = TestUtils.readBinary(out);

			TestUtils.compareMatrices(expected, result, 0.0);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

}
