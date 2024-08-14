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

package org.apache.sysds.test.functions.compress.wordembedding;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.compress.table.CompressedTableOverwriteTest;
import org.junit.Test;

public class wordEmbeddingUseCase extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(CompressedTableOverwriteTest.class.getName());

	private final static String TEST_DIR = "functions/compress/wordembedding/";

	protected String getTestClassDir() {
		return getTestDir();
	}

	protected String getTestName() {
		return "wordembedding";
	}

	protected String getTestDir() {
		return TEST_DIR;
	}

	@Test
	public void testWordEmb() {
		wordEmb(10, 2, 2, 2, ExecType.CP, "01");
	}

	@Test
	public void testWordEmb_medium() {
		wordEmb(100, 30, 4, 3, ExecType.CP, "01");
	}

	@Test
	public void testWordEmb_bigWords() {
		wordEmb(10, 2, 2, 10, ExecType.CP, "01");
	}

	@Test
	public void testWordEmb_longSentences() {
		wordEmb(100, 30, 5, 2, ExecType.CP, "01");
	}

	@Test
	public void testWordEmb_moreUniqueWordsThanSentences() {
		wordEmb(100, 200, 5, 2, ExecType.CP, "01");
	}


	public void wordEmb(int rows, int unique, int l, int embeddingSize,  ExecType instType, String name) {

		OptimizerUtils.ALLOW_SCRIPT_LEVEL_COMPRESS_COMMAND = true;
		Types.ExecMode platformOld = setExecMode(instType);

		CompressedMatrixBlock.debug = true;

		try {
			super.setOutputBuffering(true);
			loadTestConfiguration(getTestConfiguration(getTestName()));
			fullDMLScriptName = SCRIPT_DIR +  getTestClassDir() + name + ".dml";

			programArgs = new String[] {"-stats", "100", "-args", input("X"), input("W"), "" + l, output("R")};

			MatrixBlock X = TestUtils.generateTestMatrixBlock(rows, 1, 1, unique + 1, 1.0, 32);
			X = TestUtils.floor(X);
			writeBinaryWithMTD("X", X);

			MatrixBlock W = TestUtils.generateTestMatrixBlock(unique, embeddingSize, 1.0, -1, 1, 32);
			writeBinaryWithMTD("W", W);

			runTest(null);

			MatrixBlock R = TestUtils.readBinary(output("R"));

			analyzeResult(X, W, R, l);

		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	private void analyzeResult(MatrixBlock X, MatrixBlock W, MatrixBlock R, int l){
		for(int i = 0; i < X.getNumRows(); i++){
			// for each row in X, it should embed with a W, in accordance to what value it used

			// the entry to look into W. // as in row
			int e = UtilFunctions.toInt(X.get(i,0)) -1;
			int rowR = i / l;
			int offR = i % l;
			
			for(int j = 0; j < W.getNumColumns(); j++){
				assertEquals(R.get(rowR, offR* W.getNumColumns() + j), W.get(e, j), 0.0);
			}
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName()));
	}

	@Override
	protected File getConfigTemplateFile() {
		return new File("./src/test/scripts/functions/compress/SystemDS-config-compress.xml");
	}

}
