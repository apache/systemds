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

package org.apache.sysds.test.functions.builtin;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.io.IOException;
import java.util.*;

public class BuiltinTSNETest extends AutomatedTestBase
{
	private final static String TEST_NAME = "tsne";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinTSNETest.class.getSimpleName() + "/";
	
	private final static Types.ValueType[] schema = {Types.ValueType.STRING};
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"})); 
	}

	 @Test
	 public void testTSNECP() throws IOException {
		 runTSNETest(2, 30, 300.,
				 0.9, 1000, "FALSE", ExecType.CP);
	 }

	@Test
	public void testTSNESP() throws IOException {
		runTSNETest(2, 30, 300.,
				0.9, 1000, "FALSE", ExecType.SPARK);
	}

	
	private void runTSNETest(Integer reduced_dims, Integer perplexity, Double lr,
							 Double momentum, Integer max_iter, String is_verbose,	ExecType instType) throws IOException
	{
		ExecMode platformOld = setExecMode(instType);

		System.out.println("Begin t-SNE Test");
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{
				"-nvargs", "X=" + input("X"), "Y=" + output("Y"), "C=" + output("C"),
				"reduced_dims=" + reduced_dims,
				"perplexity=" + perplexity,
				"lr=" + lr,
				"momentum=" + momentum,
				"max_iter=" + max_iter,
				"is_verbose=" + is_verbose};


			double[][] A = getNonZeroRandomMatrix(200, 3, -10, 10, 7);
			writeInputMatrixWithMTD("x", A, true);

			System.out.println("Run test");
			runTest(true, false, null, -1);
			System.out.println("DONE");
			HashMap<MatrixValue.CellIndex, Double> dmlFileY = readDMLMatrixFromOutputDir("Y");
			HashMap<MatrixValue.CellIndex, Double> dmlFileC = readDMLMatrixFromOutputDir("C");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
