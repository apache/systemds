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
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class FullReblockTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "SingleReblockTest";
	private final static String TEST_NAME2 = "MultipleReblockTest";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullReblockTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rowsM = 1200;
	private final static int colsM = 1100; 
	private final static int rowsV = rowsM*colsM;
	private final static int colsV = 1; 
	private final static int blocksize = 1000; 
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.3;
	
	private static final Log LOG = LogFactory.getLog(FullReblockTest.class.getName());
	
	
	public enum Type{
		Single,
		Multiple,
		Vector
	} 
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C1", "C2" }) );
	}

	//textcell
	
	@Test
	public void testTextCellSingleMDenseCP() 
	{
		runReblockTest(FileFormat.TEXT, false, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingeMSparseCP() 
	{
		runReblockTest(FileFormat.TEXT, true, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingleVDenseCP() 
	{
		runReblockTest(FileFormat.TEXT, false, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingeVSparseCP() 
	{
		runReblockTest(FileFormat.TEXT, true, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testTextCellMultipleMDenseCP() 
	{
		runReblockTest(FileFormat.TEXT, false, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testTextCellMultipleMSparseCP() 
	{
		runReblockTest(FileFormat.TEXT, true, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingleMDenseSP() 
	{
		runReblockTest(FileFormat.TEXT, false, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellSingeMSparseSP() 
	{
		runReblockTest(FileFormat.TEXT, true, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellSingleVDenseSP() 
	{
		runReblockTest(FileFormat.TEXT, false, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellSingeVSparseSP() 
	{
		runReblockTest(FileFormat.TEXT, true, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellMultipleMDenseSP() 
	{
		runReblockTest(FileFormat.TEXT, false, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellMultipleMSparseSP() 
	{
		runReblockTest(FileFormat.TEXT, true, Type.Multiple, ExecType.SPARK);
	}
	
	//binary block
	
	@Test
	public void testBinaryBlockSingleMDenseCP() 
	{
		runReblockTest(FileFormat.BINARY, false, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingeMSparseCP() 
	{
		runReblockTest(FileFormat.BINARY, true, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingleVDenseCP() 
	{
		runReblockTest(FileFormat.BINARY, false, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingeVSparseCP() 
	{
		runReblockTest(FileFormat.BINARY, true, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockMultipleMDenseCP() 
	{
		runReblockTest(FileFormat.BINARY, false, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockMultipleMSparseCP() 
	{
		runReblockTest(FileFormat.BINARY, true, Type.Multiple, ExecType.CP);
	}
	

	@Test
	public void testBinaryBlockSingleMDenseSP() {
		runReblockTest(FileFormat.BINARY, false, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingeMSparseSP() {
		runReblockTest(FileFormat.BINARY, true, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingleVDenseSP() {
		runReblockTest(FileFormat.BINARY, false, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingeVSparseSP() {
		runReblockTest(FileFormat.BINARY, true, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockMultipleMDenseSP() {
		runReblockTest(FileFormat.BINARY, false, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockMultipleMSparseSP() {
		runReblockTest(FileFormat.BINARY, true, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingleMDenseSPAligned() {
		runReblockTest(FileFormat.BINARY, false, Type.Single, ExecType.SPARK, 500);
	}
	
	@Test
	public void testBinaryBlockSingeMSparseSPAligned() {
		runReblockTest(FileFormat.BINARY, true, Type.Single, ExecType.SPARK, 500);
	}
	
	@Test
	public void testBinaryBlockSingleVDenseSPAligned() {
		runReblockTest(FileFormat.BINARY, false, Type.Vector, ExecType.SPARK, 500);
	}
	
	@Test
	public void testBinaryBlockSingeVSparseSPAligned() {
		runReblockTest(FileFormat.BINARY, true, Type.Vector, ExecType.SPARK, 500);
	}
	
	@Test
	public void testBinaryBlockMultipleMDenseSPAligned() {
		runReblockTest(FileFormat.BINARY, false, Type.Multiple, ExecType.SPARK, 500);
	}
	
	@Test
	public void testBinaryBlockMultipleMSparseSPAligned() {
		runReblockTest(FileFormat.BINARY, true, Type.Multiple, ExecType.SPARK, 500);
	}

	private void runReblockTest( FileFormat fmt, boolean sparse, Type type, ExecType et ) {
		//force binary reblock for 999 to match 1000
		runReblockTest(fmt, sparse, type, et, blocksize-1);
	}
	
	private void runReblockTest( FileFormat fmt, boolean sparse, Type type, ExecType et, int srcBlksize )
	{
		String TEST_NAME = (type==Type.Multiple) ? TEST_NAME2 : TEST_NAME1;
		double sparsity = (sparse) ? sparsity2 : sparsity1;
		int rows = (type==Type.Vector)? rowsV : rowsM;
		int cols = (type==Type.Vector)? colsV : colsM;
		
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		if( type==Type.Multiple ) {
			programArgs = new String[]{"-args", input("A1"), input("A2"), output("C1"), output("C2")};
		}
		else {
			programArgs = new String[]{"-args", input("A"), output("C")};
		}
		
		boolean success = false;
		long seed1 = System.nanoTime();
		long seed2 = System.nanoTime()+7;

		try {
			//run test cases with single or multiple inputs
			if( type==Type.Multiple )
			{
				double[][] A1 = getRandomMatrix(rows, cols, 0, 1, sparsity, seed1);
				double[][] A2 = getRandomMatrix(rows, cols, 0, 1, sparsity, seed2);
				writeMatrix(A1, input("A1"), fmt, rows, cols, blocksize-1, blocksize-1);
				writeMatrix(A2, input("A2"), fmt, rows, cols, blocksize-1, blocksize-1);
				runTest(true, false, null, -1);
				double[][] C1 = readMatrix(output("C1"), FileFormat.BINARY, rows, cols, blocksize, blocksize);
				double[][] C2 = readMatrix(output("C2"), FileFormat.BINARY, rows, cols, blocksize, blocksize);
				TestUtils.compareMatrices(A1, C1, rows, cols, eps);
				TestUtils.compareMatrices(A2, C2, rows, cols, eps);
			}
			else {
				double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, seed1);
				writeMatrix(A, input("A"), fmt, rows, cols, blocksize-1, blocksize-1);
				runTest(true, false, null, -1);
				double[][] C = readMatrix(output("C"), FileFormat.BINARY, rows, cols, blocksize, blocksize);
				TestUtils.compareMatrices(A, C, rows, cols, eps);
			}
			
			success = true;
		}
		catch (Exception e) {
			e.printStackTrace();
			Assert.fail();
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			
			if( !success )
				LOG.error("FullReblockTest failed with seed="+seed1+", seed2="+seed2);
		}
	}
	
	private static double[][] readMatrix( String fname, FileFormat fmt, long rows, long cols, int brows, int bcols ) 
		throws IOException
	{
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, fmt, rows, cols, brows, bcols);
		double[][] C = DataConverter.convertToDoubleMatrix(mb);
		return C;
	}
	
	private static void writeMatrix( double[][] A, String fname, FileFormat fmt, long rows, long cols, int brows, int bcols ) 
		throws IOException
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, brows, bcols);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		DataConverter.writeMatrixToHDFS(mb, fname, fmt, mc);
		HDFSTool.writeMetaDataFile(fname+".mtd", ValueType.FP64, mc, fmt);
	}
}