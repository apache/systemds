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

package org.apache.sysds.test.functions.data.misc;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class NoRenameTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "NoRenameTest1";
	private final static String TEST_NAME2 = "NoRenameTest2";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + NoRenameTest.class.getSimpleName() + "/";
	
	private final static int rows = 100;
	private final static int cols = 50;
	private final static int blocksize = ConfigurationManager.getBlocksize();
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C" }) );
	}
	
	@Test
	public void testTextcellDenseSinglenode() {
		runRenameTest("text", false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextcellSparseSinglenode() {
		runRenameTest("text", true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextcsvDenseSinglenode() {
		runRenameTest("csv", false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextcsvSparseSinglenode() {
		runRenameTest("csv", true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextmmDenseSinglenode() {
		runRenameTest("mm", false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextmmSparseSinglenode() {
		runRenameTest("mm", true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextlibsvmDenseSinglenode() {
		runRenameTest("libsvm", false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTextlibsvmSparseSinglenode() {
		runRenameTest("libsvm", true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testBinaryDenseSinglenode() {
		runRenameTest("binary", false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testBinarySparseSinglenode() {
		runRenameTest("binary", true, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testTextcellDenseHybrid() {
		runRenameTest("text", false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextcellSparseHybrid() {
		runRenameTest("text", true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextcsvDenseHybrid() {
		runRenameTest("csv", false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextcsvSparseHybrid() {
		runRenameTest("csv", true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextmmDenseHybrid() {
		runRenameTest("mm", false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextmmSparseHybrid() {
		runRenameTest("mm", true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextlibsvmDenseHybrid() {
		runRenameTest("libsvm", false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextlibsvmSparseHybrid() {
		runRenameTest("libsvm", true, ExecMode.HYBRID);
	}
	
	@Test
	public void testBinaryDenseHybrid() {
		runRenameTest("binary", false, ExecMode.HYBRID);
	}
	
	@Test
	public void testBinarySparseHybrid() {
		runRenameTest("binary", true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTextcellDenseSpark() {
		runRenameTest("text", false, ExecMode.SPARK);
	}
	
	@Test
	public void testTextcellSparseSpark() {
		runRenameTest("text", true, ExecMode.SPARK);
	}
	
	@Test
	public void testTextcsvDenseSpark() {
		runRenameTest("csv", false, ExecMode.SPARK);
	}
	
	@Test
	public void testTextcsvSparseSpark() {
		runRenameTest("csv", true, ExecMode.SPARK);
	}
	
	@Test
	public void testTextmmDenseSpark() {
		runRenameTest("mm", false, ExecMode.SPARK);
	}
	
	@Test
	public void testTextmmSparseSpark() {
		runRenameTest("mm", true, ExecMode.SPARK);
	}
	
//	@Test
//	public void testTextlibsvmDenseSpark() {
//		runRenameTest("libsvm", false, ExecMode.SPARK);
//	}
//	
//	@Test
//	public void testTextlibsvmSparseSpark() {
//		runRenameTest("libsvm", true, ExecMode.SPARK);
//	}
//	
	@Test
	public void testBinaryDenseSpark() {
		runRenameTest("binary", false, ExecMode.SPARK);
	}
	
	@Test
	public void testBinarySparseSpark() {
		runRenameTest("binary", true, ExecMode.SPARK);
	}
	
	private void runRenameTest(String fmt, boolean sparse, ExecMode et)
	{
		ExecMode platformOld = setExecMode(et);
		double sparsity = (sparse) ? sparsity2 : sparsity1;
		
		String TEST_NAME = fmt.equals("binary") ? TEST_NAME2 : TEST_NAME1;
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-explain","-args", input("A"), fmt,
			String.valueOf(rows), String.valueOf(cols), output("C")};

		try {
			//write input
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(FileFormat.safeValueOf(fmt));
			writer.writeMatrixToHDFS(mbA, input("A"), rows, cols, blocksize, mbA.getNonZeros());
			
			//execute test
			runTest(true, false, null, -1);
			MatrixReader reader = MatrixReaderFactory.createMatrixReader(FileFormat.safeValueOf(fmt));
			MatrixBlock mbC = reader.readMatrixFromHDFS(output("C"), rows, cols, blocksize, -1);
			double[][] C = DataConverter.convertToDoubleMatrix(mbC);
			
			//compare matrices and check valid input
			TestUtils.compareMatrices(A, C, rows, cols, 1e-10);
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("A")));
		}
		catch(IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
