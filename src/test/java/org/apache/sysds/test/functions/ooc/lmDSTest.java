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

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.rewrite.RewriteInjectOOCTee;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.LibCommonsMath;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

public class lmDSTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "lmDS";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + lmDSTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private static final String INPUT_NAME = "X";
	private static final String INPUT_NAME2 = "y";
	private static final String OUTPUT_NAME = "R";

	private final static int rows = 10000;
	private final static int cols_wide = 500; //TODO larger than 1000
	private final static int cols_skinny = 10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void testlmDS1() {
		runMatrixVectorMultiplicationTest(cols_wide);
	}

	@Test
	public void testlmDS2() {
		runMatrixVectorMultiplicationTest(cols_skinny);
	}

	private void runMatrixVectorMultiplicationTest(int cols)
	{
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);
		boolean oldFlag = RewriteInjectOOCTee.APPLY_ONLY_XtX_PATTERN;
		
		try
		{
			RewriteInjectOOCTee.APPLY_ONLY_XtX_PATTERN = true;
			
			getAndLoadTestConfiguration(TEST_NAME1);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-ooc",
				"-args", input(INPUT_NAME), input(INPUT_NAME2), output(OUTPUT_NAME)};

			// 1. Generate the data in-memory as MatrixBlock objects
			double[][] X_data = getRandomMatrix(rows, cols, 0, 1, 1.0, 7);
			double[][] y_data = getRandomMatrix(rows, 1, 0, 1, 1.0, 3);

			// 2. Convert the double arrays to MatrixBlock objects
			MatrixBlock X_mb = DataConverter.convertToMatrixBlock(X_data);
			MatrixBlock y_mb = DataConverter.convertToMatrixBlock(y_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile
			writer.writeMatrixToHDFS(X_mb, input(INPUT_NAME), rows, cols, 1000, X_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, X_mb.getNonZeros()), Types.FileFormat.BINARY);

			// 5. Write vector x to a binary SequenceFile
			writer.writeMatrixToHDFS(y_mb, input(INPUT_NAME2), rows, 1, 1000, y_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME2 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, 1, 1000, y_mb.getNonZeros()), Types.FileFormat.BINARY);

			runTest(true, false, null, -1);
			MatrixBlock C = DataConverter.readMatrixFromHDFS(
				output(OUTPUT_NAME), Types.FileFormat.BINARY, cols, 1, 1000, 1000);
			
			//expected results
			MatrixBlock xtx = LibMatrixMult.matrixMultTransposeSelf(X_mb, new MatrixBlock(cols,cols,false), true);
			MatrixBlock xt = LibMatrixReorg.transpose(X_mb);
			MatrixBlock xty = LibMatrixMult.matrixMult(xt, y_mb);
			MatrixBlock ret = LibCommonsMath.computeSolve(xtx, xty);
			for(int i = 0; i < cols; i++)  
				Assert.assertEquals(ret.get(i, 0), C.get(i,0), eps);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
			RewriteInjectOOCTee.APPLY_ONLY_XtX_PATTERN = oldFlag;
		}
	}
}
