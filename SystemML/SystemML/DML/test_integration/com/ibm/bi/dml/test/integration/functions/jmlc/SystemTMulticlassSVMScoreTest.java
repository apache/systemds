/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.jmlc;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.jmlc.Connection;
import com.ibm.bi.dml.api.jmlc.PreparedScript;
import com.ibm.bi.dml.api.jmlc.ResultVariables;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class SystemTMulticlassSVMScoreTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "m-svm-score";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String MODEL_FILE = "sentiment_model.mtx";
	private final static double eps = 1e-10;
	
	private final static int rows = 50; //107;
	private final static int cols = 46; //fixed
	
	private final static int nRuns = 30000;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "predicted_y" })   ); 
	}

	
	@Test
	public void testJMLCMulticlassScoreDense() 
		throws IOException
	{
		//should apply diag_mm rewrite
		runJMLCMulticlassTest(false);
	}
	
	@Test
	public void testJMLCMulticlassScoreSparse() 
		throws IOException
	{
		//should apply diag_mm rewrite
		runJMLCMulticlassTest(false);
	}
	

	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 */
	private void runJMLCMulticlassTest( boolean sparse ) 
		throws IOException
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
	
		//generate inputs
		ArrayList<double[][]> Xset = generateInputs(nRuns, rows, cols, sparse?sparsity2:sparsity1); 
		
		//run DML via JMLC
		ArrayList<double[][]> Yset = execDMLScriptviaJMLC( Xset );
		
		
		//run R and compare results to DML result
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//write model data once
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(SCRIPT_DIR + TEST_DIR + MODEL_FILE, 
				                       InputInfo.TextCellInputInfo, rows, cols, 1000, 1000);
		double[][] W = DataConverter.convertToDoubleMatrix( mb );
		writeInputMatrix("W", W, true);
		
		//for each input data set
		for( int i=0; i<nRuns; i++ )
		{
			//write input data
			writeInputMatrix("X", Xset.get(i), true);	
			
			//run the R script
			runRScript(true); 
			
			//compare results
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("predicted_y");
			double[][] expected = TestUtils.convertHashMapToDoubleArray(rfile, rows, 1);
			
			TestUtils.compareMatrices(expected, Yset.get(i), rows, 1, eps);	
		}
	}

	/**
	 * 
	 * @param X
	 * @return
	 * @throws DMLException
	 * @throws IOException
	 */
	private ArrayList<double[][]> execDMLScriptviaJMLC( ArrayList<double[][]> X ) 
		throws IOException
	{
		Timing time = new Timing(true);
		
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		//establish connection to SystemML
		Connection conn = new Connection();
				
		try
		{
			//read and precompile script
			String script = conn.readScript(SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml");	
			PreparedScript pstmt = conn.prepareScript(script, new String[]{"X","W"}, new String[]{"predicted_y"});
			
			//read model
			String modelData = conn.readScript(SCRIPT_DIR + TEST_DIR + MODEL_FILE );
			double[][] W = conn.convertToDoubleMatrix(modelData, rows, cols); 
			
			//execute script multiple times
			for( int i=0; i<nRuns; i++ )
			{
				//bind input parameters
				pstmt.setMatrix("W", W);
				pstmt.setMatrix("X", X.get(i));
				
				//execute script
				ResultVariables rs = pstmt.executeScript();
				
				//get output parameter
				double[][] Y = rs.getMatrix("predicted_y");
				ret.add(Y); //keep result for comparison
			}
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new IOException(ex);
		}
		finally
		{
			if( conn != null )
				conn.close();
		}
		
		System.out.println("JMLC scoring w/ "+nRuns+" runs in "+time.stop()+"ms.");
		
		return ret;
	}
	
	/**
	 * 
	 * @param num
	 * @param rows
	 * @param cols
	 * @param sparsity
	 * @return
	 */
	private ArrayList<double[][]> generateInputs( int num, int rows, int cols, double sparsity )
	{
		ArrayList<double[][]> ret = new ArrayList<double[][]>();
		
		for( int i=0; i<num; i++ )
		{
			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, System.nanoTime());
			ret.add(X);
		}
		
		return ret;
	}
	
}