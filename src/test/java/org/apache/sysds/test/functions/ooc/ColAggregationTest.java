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

package org.apache.sysds.test.functions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class ColAggregationTest extends AutomatedTestBase{
	private static final String TEST_NAME = "ColAggregationTest";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ColAggregationTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		addTestConfiguration(TEST_NAME, config);
	}

	@Test
	public void testColAggregationNoRewrite() {
		testColAggregation(false);
	}

	/**
	 * Test the col aggregation, "colSums(X)", with OOC backend.
	 */
	@Test
	public void testColAggregationRewrite() {
		testColAggregation(true);
	}

	public void testColAggregation(boolean rewrite)
	{
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;
		boolean oldRewrite = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrite;

		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc",
				"-args", input(INPUT_NAME), output(OUTPUT_NAME)};

			int rows = 4200, cols = 2700;
			MatrixBlock mb = MatrixBlock.randOperations(rows, cols, 1.0, -1, 1, "uniform", 7);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(mb, input(INPUT_NAME), rows, cols, 1000, rows*cols);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME+".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows,cols,1000,rows*cols), Types.FileFormat.BINARY);

			runTest(true, false, null, -1);

			double[][] res = DataConverter.convertToDoubleMatrix(DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME), Types.FileFormat.BINARY, 1, cols, 1000, 1000));
			for(int j = 0; j < cols; j++) {
				double expected = 0.0;
				for(int i = 0; i < rows; i++) {
					expected += mb.get(i, j);
				}
				Assert.assertEquals(expected, res[0][j], 1e-10);
			}

			String prefix = Instruction.OOC_INST_PREFIX;
			Assert.assertTrue("OOC wasn't used for RBLK",
				heavyHittersContainsString(prefix + Opcodes.RBLK));
			// uack+
			Assert.assertTrue("OOC wasn't used for COLSUMS",
				heavyHittersContainsString(prefix + Opcodes.UACKP));
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewrite;
			resetExecMode(platformOld);
		}
	}

}
