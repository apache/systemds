/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.slowtest.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class RemoveEmptyTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "removeEmpty1";
	private final static String TEST_NAME2 = "removeEmpty2";
	private final static String TEST_NAME3 = "removeEmpty3";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int _rows = 3500;
	private final static int _cols = 2500;
	
	private final static double _sparsityDense = 0.7;
	private final static double _sparsitySparse = 0.07;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "V" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "V" })   ); 
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "V" })   ); 
	}
	
	@Test
	public void testRemoveEmptyRowsDenseCP() 
	{
		runTestRemoveEmpty( TEST_NAME1, "rows", false, false );
	}
	
	@Test
	public void testRemoveEmptyRowsSparseCP() 
	{
		runTestRemoveEmpty( TEST_NAME1, "rows", false, true );
	}
	
	@Test
	public void testRemoveEmptyColsDenseCP() 
	{
		runTestRemoveEmpty( TEST_NAME1, "cols", false, false );
	}
	
	@Test
	public void testRemoveEmptyColsSparseCP() 
	{
		runTestRemoveEmpty( TEST_NAME1, "cols", false, true );
	}

	@Test
	public void testRemoveEmptyRowsDenseMR() 
	{
		runTestRemoveEmpty( TEST_NAME1, "rows", true, false );
	}
	
	@Test
	public void testRemoveEmptyRowsSparseMR() 
	{
		runTestRemoveEmpty( TEST_NAME1, "rows", true, true );
	}
	
	@Test
	public void testRemoveEmptyColsDenseMR() 
	{
		runTestRemoveEmpty( TEST_NAME1, "cols", true, false );
	}
	
	@Test
	public void testRemoveEmptyColsSparseMR() 
	{
		runTestRemoveEmpty( TEST_NAME1, "cols", true, true );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleDenseCP() 
	{
		runTestRemoveEmpty( TEST_NAME2, "rows", false, false );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleSparseCP() 
	{
		runTestRemoveEmpty( TEST_NAME2, "rows", false, true );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleDenseCP() 
	{
		runTestRemoveEmpty( TEST_NAME2, "cols", false, false );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleSparseCP() 
	{
		runTestRemoveEmpty( TEST_NAME2, "cols", false, true );
	}

	@Test
	public void testRemoveEmptyRowsMultipleDenseMR() 
	{
		runTestRemoveEmpty( TEST_NAME2, "rows", true, false );
	}
	
	@Test
	public void testRemoveEmptyRowsMultipleSparseMR() 
	{
		runTestRemoveEmpty( TEST_NAME2, "rows", true, true );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleDenseMR() 
	{
		runTestRemoveEmpty( TEST_NAME2, "cols", true, false );
	}
	
	@Test
	public void testRemoveEmptyColsMultipleSparseMR() 
	{
		runTestRemoveEmpty( TEST_NAME2, "cols", true, true );
	}
	
	@Test
	public void testRemoveEmptyRowsDiagDenseCP() 
	{
		runTestRemoveEmpty( TEST_NAME3, "rows", false, false );
	}
	
	@Test
	public void testRemoveEmptyRowsDiagSparseCP() 
	{
		runTestRemoveEmpty( TEST_NAME3, "rows", false, true );
	}
	
	@Test
	public void testRemoveEmptyColsDiagDenseCP() 
	{
		runTestRemoveEmpty( TEST_NAME3, "cols", false, false );
	}
	
	@Test
	public void testRemoveEmptyColsDiagSparseCP() 
	{
		runTestRemoveEmpty( TEST_NAME3, "cols", false, true );
	}

	@Test
	public void testRemoveEmptyRowsDiagDenseMR() 
	{
		runTestRemoveEmpty( TEST_NAME3, "rows", true, false );
	}
	
	@Test
	public void testRemoveEmptyRowsDiagSparseMR() 
	{
		runTestRemoveEmpty( TEST_NAME3, "rows", true, true );
	}
	
	@Test
	public void testRemoveEmptyColsDiagDenseMR() 
	{
		runTestRemoveEmpty( TEST_NAME3, "cols", true, false );
	}
	
	@Test
	public void testRemoveEmptyColsDiagSparseMR() 
	{
		runTestRemoveEmpty( TEST_NAME3, "cols", true, true );
	}
	
	/**
	 * 
	 * @param testname
	 * @param margin
	 * @param mr
	 * @param sparse
	 */
	private void runTestRemoveEmpty( String testname, String margin, boolean mr, boolean sparse )
	{		
		RUNTIME_PLATFORM platformOld = rtplatform;
		if( mr )
			rtplatform = RUNTIME_PLATFORM.HADOOP; //force filebased access

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
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-args", HOME + INPUT_DIR + "V" , 
					                            String.valueOf(rows),
												String.valueOf(cols),
												margin,
												HOME + OUTPUT_DIR + "V" };
			
			loadTestConfiguration(config);
			if( cols==1 ) //test3 (removeEmpty-diag)
				createInputVector(margin, rows, sparsity);
			else //test1/test2 (general case)
				createInputMatrix(margin, rows, cols, sparsity);
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);
			
			compareResults();
		}
		finally
		{
			//reset platform for additional tests
			rtplatform = platformOld;
		}
	}

	private void createInputMatrix(String margin, int rows, int cols, double sparsity) 
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
        
        //clear out every other row/column
        if( margin.equals("rows") )
        {
        	for( int i=0; i<rows; i++ )
        	{
        		boolean clear = i%2!=0;
        		if( clear )
        			for( int j=0; j<cols; j++ )
        				V[i][j] = 0;
        		else
        			for( int j=0; j<cols; j++ )
        				Vp[i/2][j] = V[i][j];
        	}
        }
        else
        {
        	for( int j=0; j<cols; j++ )
        	{
        		boolean clear = j%2!=0;
        		if( clear )
        			for( int i=0; i<rows; i++ )
        				V[i][j] = 0;
        		else
        			for( int i=0; i<rows; i++ )
        				Vp[i][j/2] = V[i][j];
        	}
        }
        
		writeInputMatrix("V", V, false); //always text
		writeExpectedMatrix("V", Vp);
	}
	
	private void createInputVector(String margin, int rows, double sparsity) 
	{
		double[][] V = getRandomMatrix(rows, 1, 0, 1, sparsity, 7);
		double[][] Vp = null;
        
        if( margin.equals("rows") )
        {
        	int rowsp = 0;
			for(int i=0; i<rows; i++) //count nnz
				rowsp += (V[i][0]!=0)?1:0;
	        Vp = new double[rowsp][1];
        
	        for( int i=0, ix=0; i<rows; i++ )
	    		if( V[i][0]!=0 )
	    			Vp[ix++][0] = V[i][0];
        }
        else
        {
        	Vp = new double[rows][1];
        	for( int i=0; i<rows; i++ )
        		Vp[i][0] = V[i][0];	
        }
        
		writeInputMatrix("V", V, false); //always text
		writeExpectedMatrix("V", Vp);
	}
	
}