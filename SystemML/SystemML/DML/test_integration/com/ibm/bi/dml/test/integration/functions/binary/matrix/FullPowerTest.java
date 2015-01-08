/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.io.IOException;
import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullPowerTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "FullPower";
	
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1100;
	private final static int cols = 900;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	private final static double min = 0.0;
	private final static double max = 2.0;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"C"})); 
	}
	
	@Test
	public void testPowMMDenseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testPowMSDenseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testPowSMDenseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testPowSSDenseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testPowMMSparseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testPowMSSparseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, true, ExecType.CP);
	}
	
	@Test
	public void testPowSMSparseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testPowSSSparseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testPowMMDenseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testPowMSDenseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, false, ExecType.MR);
	}
	
	@Test
	public void testPowSMDenseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testPowSSDenseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, false, ExecType.MR);
	}
	
	@Test
	public void testPowMMSparseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testPowMSSparseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, true, ExecType.MR);
	}
	
	@Test
	public void testPowSMSparseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testPowSSSparseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, true, ExecType.MR);
	}
	

	/**
	 * 
	 * @param type
	 * @param dt1
	 * @param dt2
	 * @param sparse
	 * @param instType
	 */
	private void runPowerTest( DataType dt1, DataType dt2, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			String TEST_NAME = TEST_NAME1;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        HOME + INPUT_DIR + "B",
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate dataset A
			double sparsity = sparse?sparsity2:sparsity1;
			if( dt1 == DataType.MATRIX ){
				double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 7); 
				MatrixCharacteristics mcA = new MatrixCharacteristics(rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, (long) (rows*cols*sparsity));
				writeInputMatrixWithMTD("A", A, true, mcA);
			}
			else{
				double[][] A = getRandomMatrix(1, 1, min, max, 1.0, 7);
				writeScalarInputMatrixWithMTD( "A", A, true );
			}
			
			//generate dataset B
			if( dt2 == DataType.MATRIX ){
				MatrixCharacteristics mcB = new MatrixCharacteristics(rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, (long) (rows*cols*sparsity));
				double[][] B = getRandomMatrix(rows, cols, min, max, sparsity, 3); 
				writeInputMatrixWithMTD("B", B, true, mcB);
			}
			else{
				double[][] B = getRandomMatrix(1, 1, min, max, 1.0, 3);
				writeScalarInputMatrixWithMTD( "B", B, true );
			}
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = null;
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("C");
			if( dt1==DataType.SCALAR&&dt2==DataType.SCALAR )
				dmlfile = readScalarMatrixFromHDFS("C");
			else
				dmlfile = readDMLMatrixFromHDFS("C");
			
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R", true);
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	/**
	 * 
	 * @param name
	 * @param matrix
	 * @param includeR
	 */
	private void writeScalarInputMatrixWithMTD(String name, double[][] matrix, boolean includeR) 
	{
		try
		{
			//write DML scalar
			String fname = baseDirectory + INPUT_DIR + name; // + "/in";
			MapReduceTool.deleteFileIfExistOnHDFS(fname);
			MapReduceTool.writeDoubleToHDFS(matrix[0][0], fname);
			MapReduceTool.writeScalarMetaDataFile(baseDirectory + INPUT_DIR + name + ".mtd", ValueType.DOUBLE);
		
			
			//write R matrix
			if( includeR ){
				String completeRPath = baseDirectory + INPUT_DIR + name + ".mtx";
				TestUtils.writeTestMatrix(completeRPath, matrix, true);
			}
		}
		catch(IOException e)
		{
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * 
	 * @param name
	 * @return
	 */
	private HashMap<CellIndex,Double> readScalarMatrixFromHDFS(String name) 
	{
		HashMap<CellIndex,Double> dmlfile = new HashMap<CellIndex,Double>();
		try
		{
			Double val = MapReduceTool.readDoubleFromHDFSFile(baseDirectory + OUTPUT_DIR + name);
			dmlfile.put(new CellIndex(1,1), val);
		}
		catch(IOException e)
		{
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		
		return dmlfile;
	}
		
}