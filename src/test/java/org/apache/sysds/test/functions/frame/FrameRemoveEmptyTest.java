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

import static org.junit.Assert.fail;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.unary.matrix.RemoveEmptyTest;
import org.junit.Test;

public class FrameRemoveEmptyTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(FrameRemoveEmptyTest.class.getName());

	private final static String TEST_NAME1 = "removeEmpty1";
	private final static String TEST_NAME2 = "removeEmpty2";
	private final static String TEST_DIR = "functions/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RemoveEmptyTest.class.getSimpleName() + "/";

	private final static double _dense = 0.99;
	private final static double _sparse = 0.1;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
	}

	@Test
	public void testRemoveEmptyRowsCP() {
		runTestRemoveEmpty(TEST_NAME1, "rows", Types.ExecType.CP, false, false, 100, 100, _dense);
	}

	@Test
	public void testRemoveEmptyRowsCPSparse() {
		runTestRemoveEmpty(TEST_NAME1, "rows", Types.ExecType.CP, false, false, 100, 100, _sparse);
	}

	@Test
	public void testRemoveEmptyRowsCPSparse2() {
		runTestRemoveEmpty(TEST_NAME1, "rows", Types.ExecType.CP, false, false, 1000, 10, _sparse);
	}

	@Test
	public void testRemoveEmptyColsCP() {
		runTestRemoveEmpty(TEST_NAME1, "cols", Types.ExecType.CP, false, false, 100, 100, _dense);
	}

	@Test
	public void testRemoveEmptyColsCPSparse() {
		runTestRemoveEmpty(TEST_NAME1, "cols", Types.ExecType.CP, false, false, 100, 100, _sparse);
	}

	@Test
	public void testRemoveEmptyColsCPSparse2() {
		runTestRemoveEmpty(TEST_NAME1, "cols", Types.ExecType.CP, false, false, 10, 1000, _sparse);
	}

	@Test
	public void testRemoveEmptyRowsSelectFullCP() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, true, 100, 100, _dense);
	}

	@Test
	public void testRemoveEmptyRowsSelectFullCPSparse() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, true, 100, 100, _sparse);
	}

	@Test
	public void testRemoveEmptyRowsSelectFullCPSparse2() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, true, 100, 10, _sparse);
	}

	@Test
	public void testRemoveEmptyColsSelectFullCP() {
		runTestRemoveEmpty(TEST_NAME2, "cols", Types.ExecType.CP, true, true, 100, 100, _dense);
	}

	@Test
	public void testRemoveEmptyColsSelectFullCPSparse() {
		runTestRemoveEmpty(TEST_NAME2, "cols", Types.ExecType.CP, true, true, 100, 100, _sparse);
	}

	@Test
	public void testRemoveEmptyRowsSelectCP() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, false, 100, 100, _dense);
	}

	@Test
	public void testRemoveEmptyRowsSelectCPSparse() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, false, 100, 100, _sparse);
	}

	@Test
	public void testRemoveEmptyRowsSelectCPSparse2() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, false, 100, 10, _sparse);
	}

	@Test
	public void testRemoveEmptyRowsSelectCPSparse3() {
		runTestRemoveEmpty(TEST_NAME2, "rows", Types.ExecType.CP, true, false, 100, 3, _sparse);
	}

	@Test
	public void testRemoveEmptyColsSelectCP() {
		runTestRemoveEmpty(TEST_NAME2, "cols", Types.ExecType.CP, true, false, 100, 100, _dense);
	}

	@Test
	public void testRemoveEmptyColsSelectCPSparse() {
		runTestRemoveEmpty(TEST_NAME2, "cols", Types.ExecType.CP, true, false, 100, 100, _sparse);
	}

	private void runTestRemoveEmpty(String testname, String margin, Types.ExecType et, boolean bSelectIndex,
		boolean fullSelect, int rows, int cols, double sparsity) {
		Types.ExecMode platformOld = rtplatform;
		switch(et) {
			case SPARK:
				rtplatform = Types.ExecMode.SPARK;
				break;
			default:
				rtplatform = Types.ExecMode.HYBRID;
				break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == Types.ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		setOutputBuffering(true);
		try {
			// register test configuration
			TestConfiguration config = getTestConfiguration(testname);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[] {"-explain", "-args", input("V"), input("I"), margin, output("R")};

			Pair<MatrixBlock, MatrixBlock> data = createInputMatrix(margin, bSelectIndex, fullSelect, rows, cols, sparsity);

			MatrixBlock in = data.getKey();
			MatrixBlock select = data.getValue();

			runTest(null);

			MatrixBlock expected = fullSelect ? in : in.removeEmptyOperations(new MatrixBlock(), margin.equals("rows"),
				false, select);

			double[][] out = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("R"));

			LOG.debug(expected.getNumRows() + "  " + out.length);

			TestUtils.compareMatrices(expected, out, 0, "");
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed test because of exception " + e);
		}
		finally {
			// reset platform for additional tests
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	private Pair<MatrixBlock, MatrixBlock> createInputMatrix(String margin, boolean bSelectIndex, boolean fullSelect,
		int rows, int cols, double sparsity) {
		int rowsp = -1, colsp = -1;
		if(margin.equals("rows")) {
			rowsp = rows / 2;
			colsp = cols;
		}
		else {
			rowsp = rows;
			colsp = cols / 2;
		}

		// long seed = System.nanoTime();
		double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
		double[][] Vp = new double[rowsp][colsp];
		double[][] Ix;
		int innz = 0, vnnz = 0;

		// clear out every other row/column
		if(margin.equals("rows")) {
			Ix = new double[rows][1];
			for(int i = 0; i < rows; i++) {
				boolean clear = i % 2 != 0;
				if(clear && !fullSelect) {
					for(int j = 0; j < cols; j++)
						V[i][j] = 0;
					Ix[i][0] = 0;
				}
				else {
					boolean bNonEmpty = false;
					for(int j = 0; j < cols; j++) {
						Vp[i / 2][j] = V[i][j];
						bNonEmpty |= V[i][j] != 0.0;
						vnnz += (V[i][j] == 0.0) ? 0 : 1;
					}
					Ix[i][0] = (bNonEmpty || fullSelect) ? 1 : 0;
					innz += Ix[i][0];
				}
			}
		}
		else {
			Ix = new double[1][cols];
			for(int j = 0; j < cols; j++) {
				boolean clear = j % 2 != 0;
				if(clear && !fullSelect) {
					for(int i = 0; i < rows; i++)
						V[i][j] = 0;
					Ix[0][j] = 0;
				}
				else {
					boolean bNonEmpty = false;
					for(int i = 0; i < rows; i++) {
						Vp[i][j / 2] = V[i][j];
						bNonEmpty |= V[i][j] != 0.0;
						vnnz += (V[i][j] == 0.0) ? 0 : 1;
					}
					Ix[0][j] = (bNonEmpty || fullSelect) ? 1 : 0;
					innz += Ix[0][j];
				}
			}
		}

		MatrixCharacteristics imc = new MatrixCharacteristics(margin.equals("rows") ? rows : 1,
			margin.equals("rows") ? 1 : cols, 1000, innz);
		MatrixCharacteristics vmc = new MatrixCharacteristics(rows, cols, 1000, vnnz);

		MatrixBlock in = new MatrixBlock(rows, cols, false);
		in.init(V, rows, cols);

		MatrixBlock select = new MatrixBlock(Ix.length, Ix[0].length, false);
		select.init(Ix, Ix.length, Ix[0].length);

		writeInputMatrixWithMTD("V", V, false, vmc); // always text
		writeExpectedMatrix("V", Vp);
		if(bSelectIndex)
			writeInputMatrixWithMTD("I", Ix, false, imc);

		in.examSparsity();
		select.examSparsity();

		return new ImmutablePair<>(in, select);
	}
}
