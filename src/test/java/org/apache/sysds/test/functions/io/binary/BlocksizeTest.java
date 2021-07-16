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

package org.apache.sysds.test.functions.io.binary;

import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BlocksizeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "BlocksizeTest";
	private final static String TEST_DIR = "functions/io/binary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BlocksizeTest.class.getSimpleName() + "/";
	
	public static int rows = 2345;
	public static int cols = 4321;
	public static double sparsity = 0.05;
	private final static double eps = 1e-14;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "X" }) );  
	}
	
	@Test
	public void testSingleNode1000_1000() {
		runBlocksizeTest(1000, 1000, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testHybrid1000_1000() {
		runBlocksizeTest(1000, 1000, ExecMode.HYBRID);
	}
	
	@Test
	public void testSpark1000_1000() {
		runBlocksizeTest(1000, 1000, ExecMode.SPARK);
	}
	
	@Test
	public void testSingleNode1006_1000() {
		runBlocksizeTest(1006, 1000, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testHybrid1006_1000() {
		runBlocksizeTest(1006, 1000, ExecMode.HYBRID);
	}
	
	@Test
	public void testSpark1006_1000() {
		runBlocksizeTest(1006, 1000, ExecMode.SPARK);
	}
	
	@Test
	public void testSingleNode1006_503() {
		runBlocksizeTest(1006, 503, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testHybrid1006_503() {
		runBlocksizeTest(1006, 503, ExecMode.HYBRID);
	}
	
	@Test
	public void testSpark1006_503() {
		runBlocksizeTest(1006, 503, ExecMode.SPARK);
	}

	@Test
	public void testSingleNode2000() {
		runBlocksizeTest(1000, 2000, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testHybrid2000() {
		runBlocksizeTest(1000, 2000, ExecMode.HYBRID);
	}
	
	@Test
	public void testSpark2000() {
		runBlocksizeTest(1000, 2000, ExecMode.SPARK);
	}
	
	@Test
	public void testSingleNode2xRowsCols() {
		runBlocksizeTest(1000, 7000, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testHybrid2xRowsCols() {
		runBlocksizeTest(1000, 7000, ExecMode.HYBRID);
	}
	
	@Test
	public void testSpark2xRowsCols() {
		//test for invalid shuffle-free reblock
		runBlocksizeTest(1000, 7000, ExecMode.SPARK);
	}
	
	private void runBlocksizeTest(int inBlksize, int outBlksize, ExecMode mode)
	{
		ExecMode modeOld = setExecMode(mode);
		try {
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args",
				input("X"), output("X"), String.valueOf(outBlksize)};
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(rows, cols, -1.0, 1.0, sparsity, 7); 
			MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, inBlksize);
			DataConverter.writeMatrixToHDFS(mb, input("X"), FileFormat.BINARY, mc);
			HDFSTool.writeMetaDataFile(input("X.mtd"), ValueType.FP64, mc, FileFormat.BINARY);
			
			runTest(true, false, null, -1); //mult 7
			
			//compare matrices 
			checkDMLMetaDataFile("X", new MatrixCharacteristics(rows, cols, outBlksize), true);
			MatrixBlock mb2 = DataConverter.readMatrixFromHDFS(
				output("X"), FileFormat.BINARY, rows, cols, outBlksize, -1);
			for( int i=0; i<mb.getNumRows(); i++ )
				for( int j=0; j<mb.getNumColumns(); j++ ) {
					double val1 = mb.quickGetValue(i, j) * 7;
					double val2 = mb2.quickGetValue(i, j);
					Assert.assertEquals(val1, val2, eps);
				}
		}
		catch(IOException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}
