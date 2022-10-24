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

package org.apache.sysds.test.functions.unary.matrix;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class RemoveEmptyTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "removeEmpty1";
	private final static String TEST_NAME2 = "removeEmpty2";
	private final static String TEST_NAME3 = "removeEmpty3";
	private final static String TEST_NAME4 = "removeEmpty4";
	
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RemoveEmptyTest.class.getSimpleName() + "/";

	private final static int _rows = 2500;
	private final static int _cols = 1500;
	
	private final static double _sparsityDense = 0.7;
	private final static double _sparsitySparse = 0.07;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "V" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "V" }) );
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "V" }) );
		addTestConfiguration(TEST_NAME4, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "V" }) );
	}
	
	@Test
	public void testRemoveEmptyRowsDenseCP() {
		runTestRemoveEmpty( TEST_NAME1, "rows", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyRowsSparseCP() {
		runTestRemoveEmpty( TEST_NAME1, "rows", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyColsDenseCP() {
		runTestRemoveEmpty( TEST_NAME1, "cols", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyColsSparseCP() {
		runTestRemoveEmpty( TEST_NAME1, "cols", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyRowsDenseSP() {
		runTestRemoveEmpty( TEST_NAME1, "rows", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyRowsSparseSP() {
		runTestRemoveEmpty( TEST_NAME1, "rows", ExecType.SPARK, true );
	}
	
	@Test
	public void testRemoveEmptyColsDenseSP() {
		runTestRemoveEmpty( TEST_NAME1, "cols", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyColsSparseSP() {
		runTestRemoveEmpty( TEST_NAME1, "cols", ExecType.SPARK, true );
	}

	@Test
	public void testRemoveEmptyRowsMultipleDenseCP() {
		runTestRemoveEmpty( TEST_NAME2, "rows", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleSparseCP() {
		runTestRemoveEmpty( TEST_NAME2, "rows", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleDenseCP() {
		runTestRemoveEmpty( TEST_NAME2, "cols", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleSparseCP() {
		runTestRemoveEmpty( TEST_NAME2, "cols", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleDenseSP() {
		runTestRemoveEmpty( TEST_NAME2, "rows", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleSparseSP() {
		runTestRemoveEmpty( TEST_NAME2, "rows", ExecType.SPARK, true );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleDenseSP() {
		runTestRemoveEmpty( TEST_NAME2, "cols", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleSparseSP() {
		runTestRemoveEmpty( TEST_NAME2, "cols", ExecType.SPARK, true );
	}
	
	@Test
	public void testRemoveEmptyRowsDiagDenseCP() {
		runTestRemoveEmpty( TEST_NAME3, "rows", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyRowsDiagSparseCP() {
		runTestRemoveEmpty( TEST_NAME3, "rows", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyColsDiagDenseCP() {
		runTestRemoveEmpty( TEST_NAME3, "cols", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyColsDiagSparseCP() {
		runTestRemoveEmpty( TEST_NAME3, "cols", ExecType.CP, true );
	}

	@Test
	public void testRemoveEmptyRowsDiagDenseSP() {
		runTestRemoveEmpty( TEST_NAME3, "rows", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyRowsDiagSparseSP() {
		runTestRemoveEmpty( TEST_NAME3, "rows", ExecType.SPARK, true );
	}
	
	@Test
	public void testRemoveEmptyColsDiagDenseSP() {
		runTestRemoveEmpty( TEST_NAME3, "cols", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyColsDiagSparseSP() {
		runTestRemoveEmpty( TEST_NAME3, "cols", ExecType.SPARK, true );
	}
	
	// ------------------------------
	// Remove Empty with Broadcast option
	@Test
	public void testRemoveEmptyRowsDenseBcSP() {
		runTestRemoveEmpty( TEST_NAME1, "rows", ExecType.SPARK, false, false );
	}
	
	@Test
	public void testRemoveEmptyRowsSparseBcSP() {
		runTestRemoveEmpty( TEST_NAME1, "rows", ExecType.SPARK, true, false );
	}
	
	@Test
	public void testRemoveEmptyColsDenseBcSP() {
		runTestRemoveEmpty( TEST_NAME1, "cols", ExecType.SPARK, false, false );
	}
	
	@Test
	public void testRemoveEmptyColsSparseBcSP() {
		runTestRemoveEmpty( TEST_NAME1, "cols", ExecType.SPARK, true, false );
	}

	@Test
	public void testRemoveEmptyRowsMultipleDenseBcSP() {
		runTestRemoveEmpty( TEST_NAME2, "rows", ExecType.SPARK, false, false );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleSparseBcSP() {
		runTestRemoveEmpty( TEST_NAME2, "rows", ExecType.SPARK, true, false );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleDenseBcSP() {
		runTestRemoveEmpty( TEST_NAME2, "cols", ExecType.SPARK, false, false );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleSparseBcSP() {
		runTestRemoveEmpty( TEST_NAME2, "cols", ExecType.SPARK, true, false );
	}
	
	//-------------------------------------------------------------------------------------
	//  Testcases to pass index containing non-empty rows.
	// CP Test cases
	@Test
	public void testRemoveEmptyRowsDenseCPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "rows", ExecType.CP, false, true, true);
	}
	
	@Test
	public void testRemoveEmptyColsDenseCPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "cols", ExecType.CP, false, true, true);
	}
	
	@Test
	public void testRemoveEmptyRowsSparseCPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "rows", ExecType.CP, true, true, true);
	}
	
	@Test
	public void testRemoveEmptyColsSparseCPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "cols", ExecType.CP, true, true, true);
	}
	
	// Spark Test cases
	@Test
	public void testRemoveEmptyRowsDenseSPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "rows", ExecType.SPARK, false, true, true);
	}
	
	@Test
	public void testRemoveEmptyColsDenseSPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "cols", ExecType.SPARK, false, true, true);
	}
	
	@Test
	public void testRemoveEmptyRowsSparseSPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "rows", ExecType.SPARK, true, true, true);
	}
	
	@Test
	public void testRemoveEmptyColsSparseSPwIdx() {
		runTestRemoveEmpty( TEST_NAME4, "cols", ExecType.SPARK, true, true, true);
	}
	
	private void runTestRemoveEmpty( String testname, String margin, ExecType et, boolean sparse ) {
		runTestRemoveEmpty(testname, margin, et, sparse, true);
	}

	private void runTestRemoveEmpty( String testname, String margin, ExecType et, boolean sparse, boolean bForceDistRmEmpty) {
		runTestRemoveEmpty(testname, margin, et, sparse, bForceDistRmEmpty, false);
	}
	
	private void runTestRemoveEmpty( String testname, String margin, ExecType et, boolean sparse, boolean bForceDistRmEmpty, boolean bSelectIndex)
	{
		ExecMode platformOld = setExecMode(et);
		ParameterizedBuiltinOp.FORCE_DIST_RM_EMPTY = bForceDistRmEmpty;
		
		try
		{
			//setup dims and sparsity
			int rows = _rows;
			int cols = (testname.equals(TEST_NAME3))? 1 : _cols;
			double sparsity = sparse ? _sparsitySparse : _sparsityDense;
				
			//register test configuration
			TestConfiguration config = getTestConfiguration(testname);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			if (!bSelectIndex) {
				if(!testname.equals(TEST_NAME3))
					programArgs = new String[]{"-explain", "-args", input("V"), 
						margin, output("V") };
				else
					programArgs = new String[]{"-explain", "-args", input("V"), 
						String.valueOf(rows), String.valueOf(cols), margin, output("V") };
			}
			else
				programArgs = new String[]{"-explain", "-args", input("V"), input("I"),
					margin, output("V") };
			
			if( cols==1 ) //test3 (removeEmpty-diag)
				createInputVector(margin, rows, sparsity, bSelectIndex);
			else //test1/test2 (general case)
				createInputMatrix(margin, rows, cols, sparsity, bSelectIndex);
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			
			compareResults();
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private void createInputMatrix(String margin, int rows, int cols, double sparsity, boolean bSelectIndex) 
	{
		int rowsp = -1, colsp = -1;
		if( margin.equals("rows") ){
			rowsp = rows/2;
			colsp = cols;
		}
		else {
			rowsp = rows;
			colsp = cols/2;
		}
		
		//long seed = System.nanoTime();
		double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
		double[][] Vp = new double[rowsp][colsp];
		double[][] Ix = null;
		int innz = 0, vnnz = 0;
		
		//clear out every other row/column
		if( margin.equals("rows") )
		{
			Ix = new double[rows][1];
			for( int i=0; i<rows; i++ )
			{
				boolean clear = i%2!=0;
				if( clear ) {
					for( int j=0; j<cols; j++ )
						V[i][j] = 0;
					Ix[i][0] = 0;
				}
				else {
					boolean bNonEmpty = false;
					for( int j=0; j<cols; j++ )
					{
						Vp[i/2][j] = V[i][j];
						bNonEmpty |= (V[i][j] != 0.0)?true:false;
						vnnz += (V[i][j] == 0.0)?0:1;
					}
					Ix[i][0] = (bNonEmpty)?1:0;
					innz += Ix[i][0]; 
				}
			}
		}
		else
		{
			Ix = new double[1][cols];
			for( int j=0; j<cols; j++ )
			{
				boolean clear = j%2!=0;
				if( clear ) {
					for( int i=0; i<rows; i++ )
						V[i][j] = 0;
					Ix[0][j] = 0;
				}
				else {
					boolean bNonEmpty = false;
					for( int i=0; i<rows; i++ ) 
					{
						Vp[i][j/2] = V[i][j];
						bNonEmpty |= (V[i][j] != 0.0)?true:false;
						vnnz += (V[i][j] == 0.0)?0:1;
					}
					Ix[0][j] = (bNonEmpty)?1:0;
					innz += Ix[0][j]; 
				}
			}
		}
		
		MatrixCharacteristics imc = new MatrixCharacteristics(margin.equals("rows")?rows:1, margin.equals("rows")?1:cols, 1000, innz);
		MatrixCharacteristics vmc = new MatrixCharacteristics(rows, cols, 1000, vnnz);
		
		writeInputMatrixWithMTD("V", V, false, vmc); //always text
		writeExpectedMatrix("V", Vp);
		if(bSelectIndex)
			writeInputMatrixWithMTD("I", Ix, false, imc);
	}
	
	private void createInputVector(String margin, int rows, double sparsity, boolean bSelectIndex) 
	{
		double[][] V = getRandomMatrix(rows, 1, 0, 1, sparsity, 7);
		double[][] Vp = null;
		double[][] Ix = new double[rows][1];
		
		if( margin.equals("rows") )
		{
			int rowsp = 0;
			for(int i=0; i<rows; i++) //count nnz
				rowsp += (V[i][0]!=0)?1:0;
			Vp = new double[rowsp][1];
		
			for( int i=0, ix=0; i<rows; i++ )
				if( V[i][0]!=0 ) {
					Vp[ix++][0] = V[i][0];
					Ix[i][0] = 1;
				} else
					Ix[i][0] = 0;
		}
		else
		{
			Vp = new double[rows][1];
			for( int i=0; i<rows; i++ ) {
				Vp[i][0] = V[i][0];	
				if( V[i][0]!=0 ) {
					Ix[i][0] = 1;
				} else
					Ix[i][0] = 0;
			}
		}
		
		writeInputMatrix("V", V, false); //always text
		writeExpectedMatrix("V", Vp);
		if(bSelectIndex)
			writeInputMatrix("I", Ix, false);
	}
}
