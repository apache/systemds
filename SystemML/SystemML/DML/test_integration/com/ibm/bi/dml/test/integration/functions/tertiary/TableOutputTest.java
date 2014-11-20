/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.tertiary;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class TableOutputTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "TableOutputTest";
	
	private final static String TEST_DIR = "functions/tertiary/";
	
	private final static int rows = 50000;
	private final static int maxVal1 = 7, maxVal2 = 15; 
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "F" })   ); 
	}

	
	@Test
	public void testTableOutputCP1() 
	{
		runTableOutputTest(ExecType.CP, 0);
	}
	
	@Test
	public void testTableOutputCP2() 
	{
		runTableOutputTest(ExecType.CP, 5);
	}
	
	@Test
	public void testTableOutputCP3() 
	{
		runTableOutputTest(ExecType.CP, -5);
	}
	
	@Test
	public void testTableOutputMR1() 
	{
		runTableOutputTest(ExecType.MR, 0);
	}
	
	@Test
	public void testTableOutputMR2() 
	{
		runTableOutputTest(ExecType.MR, 5);
	}
	
	@Test
	public void testTableOutputMR3() 
	{
		runTableOutputTest(ExecType.MR, -5);
	}
	

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runTableOutputTest( ExecType et, int delta)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;

		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);

			int dim1 = maxVal1 + delta;
			int dim2 = maxVal2 + delta;

			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", 
											HOME + INPUT_DIR + "A", 
											HOME + INPUT_DIR + "B",
					                        Integer.toString(dim1),
					                        Integer.toString(dim2),
					                        HOME + OUTPUT_DIR + "F"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
			
			//generate actual dataset (always dense because values <=0 invalid)
			double[][] A = floor(getRandomMatrix(rows, 1, 1, maxVal1, 1.0, -1), rows, 1); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = floor(getRandomMatrix(rows, 1, 1, maxVal2, 1.0, -1), rows, 1); 
			writeInputMatrixWithMTD("B", B, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("F");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("F");
			
			CellIndex tmp = new CellIndex(-1, -1);
			double dmlVal, rVal;
			int numErrors = 0;
			
			try {
			for(int i=1; i<=Math.min(dim1, maxVal1); i++) {
				for(int j=1; j<=Math.min(dim2, maxVal2); j++) {
					tmp.set(i, j);
					dmlVal = ( dmlfile.get(tmp) == null ? 0 : dmlfile.get(tmp) );
					rVal = ( rfile.get(tmp) == null ? 0 : rfile.get(tmp) );
					if ( dmlVal != rVal ) {
						System.err.println("  (" + i+","+j+ ") " + dmlVal + " != " + rVal);
						numErrors++;
					}
				}
			}
			} catch(Exception e) {
				e.printStackTrace();
			}
			Assert.assertEquals(0, numErrors);

			numErrors = 0;
			if ( delta > 0 ) {
				// check for correct padding in dmlfile
				for(int i=1; i<= delta; i++) {
					for(int j=1; j<=delta; j++) {
						tmp.set(maxVal1+i, maxVal2+j);
						dmlVal = ( dmlfile.get(tmp) == null ? 0 : dmlfile.get(tmp) );
						if(dmlVal != 0) {
							System.err.println("  Padding: (" + i+","+j+ ") " + dmlVal + " != 0");
							numErrors++;
						}
					}
				}
			}
			Assert.assertEquals(0, numErrors);
			
			
		}
		finally
		{
			rtplatform = platformOld;
		}
	}

	/**
	 * 
	 * @param X
	 * @param rows
	 * @param cols
	 * @return
	 */
	private double[][] floor( double[][] X, int rows, int cols )
	{
		for( int i=0; i<rows; i++ )
			for( int j=0; j<cols; j++ )
				X[i][j] = Math.floor(X[i][j]);
		return X;
	}
} 