/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.caching;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class CachingPWriteExportTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "export";
	private final static String TEST_DIR = "functions/caching/";

	private final static int rows = (int)Hop.CPThreshold-1;
	private final static int cols = (int)Hop.CPThreshold-1;    
	private final static double sparsity = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "V" })   ); 
	}
	
	@Test
	public void testExportReadWrite() 
	{
		runTestExport( "binary" );
	}
	
	@Test
	public void testExportCopy() 
	{
		runTestExport( "text" );
	}

	
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runTestExport( String outputFormat )
	{				
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "V",
				                        outputFormat };
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
		writeInputMatrix("V", V, true); //always text
		writeExpectedMatrix("V", V);
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		
		double[][] Vp = null;
		try
		{
			InputInfo ii = null;
			if( outputFormat.equals("binary") )
				ii = InputInfo.BinaryBlockInputInfo;
			else
				ii = InputInfo.TextCellInputInfo;
			
			MatrixBlock mb = DataConverter.readMatrixFromHDFS(HOME + OUTPUT_DIR + "V", ii, rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, sparsity);
			Vp = DataConverter.convertToDoubleMatrix(mb);
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);		}
		
		//compare
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				if( V[i][j]!=Vp[i][j] )
					//System.out.println("Wrong value i="+i+", j="+j+", value1="+V[i][j]+", value2="+Vp[i][j]);
					Assert.fail("Wrong value i="+i+", j="+j+", value1="+V[i][j]+", value2="+Vp[i][j]);
	}
}