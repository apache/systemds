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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;

import static org.apache.sysds.test.TestUtils.readDMLMatrixFromHDFS;

public class UnaryTest extends AutomatedTestBase {

	private static final String TEST_NAME = "Unary";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + UnaryTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		addTestConfiguration(TEST_NAME, config);
	}

	/**
	 * Test the sum of scalar multiplication, "sum(X*7)", with OOC backend.
	 */
	@Test
	public void testUnary() {
		testUnaryOperation(false);
	}
	
	
	public void testUnaryOperation(boolean rewrite)
	{
		Types.ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);
		boolean oldRewrite = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrite;
		
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", 
				"-args", input(INPUT_NAME), output(OUTPUT_NAME)};

			int rows = 1000, cols = 1000;
			MatrixBlock mb = MatrixBlock.randOperations(rows, cols, 1.0, -1, 1, "uniform", 7);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(FileFormat.BINARY);
			writer.writeMatrixToHDFS(mb, input(INPUT_NAME), rows, cols, 1000, rows*cols);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME+".mtd"), ValueType.FP64, 
				new MatrixCharacteristics(rows,cols,1000,rows*cols), FileFormat.BINARY);
			
			runTest(true, false, null, -1);

			double[][] C1 = readMatrix(output(OUTPUT_NAME), FileFormat.BINARY, rows, cols, 1000, 1000);
			double expected = 0.0;
			double result = 0.0;
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					expected = Math.ceil(mb.get(i, j));
					result = C1[i][j];
					Assert.assertEquals(expected, result, 1e-10);
				}
			}

			String prefix = Instruction.OOC_INST_PREFIX;
			Assert.assertTrue("OOC wasn't used for RBLK",
				heavyHittersContainsString(prefix + Opcodes.RBLK));
			Assert.assertTrue("OOC wasn't used for CEIL",
				heavyHittersContainsString(prefix + Opcodes.CEIL));
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewrite;
			resetExecMode(platformOld);
		}
	}

	private static double[][] readMatrix( String fname, FileFormat fmt, long rows, long cols, int brows, int bcols )
			throws IOException
	{
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, fmt, rows, cols, brows, bcols);
		double[][] C = DataConverter.convertToDoubleMatrix(mb);
		return C;
	}
}
