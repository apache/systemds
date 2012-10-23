package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class RemoveEmptyTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "removeEmpty";
	private final static String TEST_DIR = "functions/unary/matrix/";

	//note: even number of rows/cols required.
	private final static int rowsSmall = 3500;
	private final static int colsSmall = 2500;
	private final static int rowsLarge = 3500;
	private final static int colsLarge = 2500;
	
	private final static double sparsityDense = 0.7;
	private final static double sparsitySparse = 0.07;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "V" })   ); 
	}
	
	@Test
	public void testRemoveEmptyRowsSmallDense() 
	{
		runTestRemoteEmpty( "rows", false, false );
	}
	
	@Test
	public void testRemoveEmptyRowsSmallSparse() 
	{
		runTestRemoteEmpty( "rows", false, true );
	}
	
	@Test
	public void testRemoveEmptyColsSmallDense() 
	{
		runTestRemoteEmpty( "cols", false, false );
	}
	
	@Test
	public void testRemoveEmptyColsSmallSparse() 
	{
		runTestRemoteEmpty( "cols", false, true );
	}

	@Test
	public void testRemoveEmptyRowsLargeDense() 
	{
		runTestRemoteEmpty( "rows", true, false );
	}
	
	@Test
	public void testRemoveEmptyRowsLargeSparse() 
	{
		runTestRemoteEmpty( "rows", true, true );
	}
	
	@Test
	public void testRemoveEmptyColsLargeDense() 
	{
		runTestRemoteEmpty( "cols", true, false );
	}
	
	@Test
	public void testRemoveEmptyColsLargeSparse() 
	{
		runTestRemoteEmpty( "cols", true, true );
	}
	

	private void runTestRemoteEmpty( String margin, boolean large, boolean sparse )
	{		
		//handle rows and cols
		RUNTIME_PLATFORM platformOld = rtplatform;
		
		int rows = -1, cols = -1;
		if( large ){
			rows = rowsLarge;
			cols = colsLarge;
			
			rtplatform = RUNTIME_PLATFORM.HADOOP; //force filebased access
		}
		else{
			rows = rowsSmall;
			cols = colsSmall;
		}
		
		//handle sparsity
		double sparsity = -1;
		if( sparse )
			sparsity = sparsitySparse;
		else
			sparsity = sparsityDense;
			
		//register test configuration
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                            String.valueOf(rows),
											String.valueOf(cols),
											margin,
											HOME + OUTPUT_DIR + "V" };
		
		loadTestConfiguration(config);
		createSmallInputMatrix(margin, rows, cols, sparsity);
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		
		compareResults();
		
		//reset platform for additional tests
		rtplatform = platformOld;
	}

	private void createSmallInputMatrix(String margin, int rows, int cols, double sparsity) 
	{
		int rowsp = -1, colsp = -1;
		if( margin.equals("rows") ){
			rowsp = rows/2;
			colsp = cols;
		}
		else{
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
	
	private void createLargeInputMatrix(String margin, int rows, int cols, double sparsity) 
	{
		
		int rowsp = -1, colsp = -1;
		if( margin.equals("rows") ){
			rowsp = rows/2;
			colsp = cols;
		}
		else{
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
	
}