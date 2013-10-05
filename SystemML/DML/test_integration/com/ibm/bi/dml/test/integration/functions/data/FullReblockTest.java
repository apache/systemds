/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.data;

import java.io.IOException;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullReblockTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "SingleReblockTest";
	private final static String TEST_NAME2 = "MultipleReblockTest";
	private final static String TEST_DIR = "functions/data/";
	private final static double eps = 1e-10;
	
	private final static int rowsM = 1200;
	private final static int colsM = 1100; 
	private final static int rowsV = rowsM*colsM;
	private final static int colsV = 1; 
	private final static int blocksize = 1000; 
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.3;
	
	public enum Type{
		Single,
		Multiple,
		Vector
	} 
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "C" })   );
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "C1", "C2" })   );
	}

	
	@Test
	public void testTextCellSingleMDenseCP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingeMSparseCP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingleVDenseCP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingeVSparseCP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testTextCellMultipleMDenseCP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testTextCellMultipleMSparseCP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testTextCellSingleMDenseMR() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Single, ExecType.MR);
	}
	
	@Test
	public void testTextCellSingeMSparseMR() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Single, ExecType.MR);
	}
	
	
	@Test
	public void testTextCellSingleVDenseMR() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Vector, ExecType.MR);
	}
	
	@Test
	public void testTextCellSingeVSparseMR() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Vector, ExecType.MR);
	}
	
	@Test
	public void testTextCellMultipleMDenseMR() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Multiple, ExecType.MR);
	}
	
	@Test
	public void testTextCellMultipleMSparseMR() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Multiple, ExecType.MR);
	}
	
	//binary block
	
	@Test
	public void testBinaryBlockSingleMDenseCP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingeMSparseCP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingleVDenseCP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingeVSparseCP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockMultipleMDenseCP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockMultipleMSparseCP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testBinaryBlockSingleMDenseMR() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Single, ExecType.MR);
	}
	
	@Test
	public void testBinaryBlockSingeMSparseMR() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Single, ExecType.MR);
	}
	
	@Test
	public void testBinaryBlockSingleVDenseMR() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Vector, ExecType.MR);
	}
	
	@Test
	public void testBinaryBlockSingeVSparseMR() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Vector, ExecType.MR);
	}
	
	@Test
	public void testBinaryBlockMultipleMDenseMR() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Multiple, ExecType.MR);
	}
	
	@Test
	public void testBinaryBlockMultipleMSparseMR() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Multiple, ExecType.MR);
	}
	
	/*
	@Test
	public void testBinaryCellSingleDenseCP() 
	{
		runReblockTest(OutputInfo.BinaryCellOutputInfo, false, false, ExecType.CP);
	}
	
	@Test
	public void testBinaryCellSingeSparseCP() 
	{
		runReblockTest(OutputInfo.BinaryCellOutputInfo, true, false, ExecType.CP);
	}
	
	@Test
	public void testBinaryCellMultipleDenseCP() 
	{
		runReblockTest(OutputInfo.BinaryCellOutputInfo, false, true, ExecType.CP);
	}
	
	@Test
	public void testBinaryCellMultipleSparseCP() 
	{
		runReblockTest(OutputInfo.BinaryCellOutputInfo, true, true, ExecType.CP);
	}
	 */
	
	private void runReblockTest( OutputInfo oi, boolean sparse, Type type, ExecType et )
	{		
		String TEST_NAME = (type==Type.Multiple) ? TEST_NAME2 : TEST_NAME1;		 
		double sparsity = (sparse) ? sparsity2 : sparsity1;		
		int rows = (type==Type.Vector)? rowsV : rowsM;
		int cols = (type==Type.Vector)? colsV : colsM;
		
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		if( type==Type.Multiple ) {
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A1",
												HOME + INPUT_DIR + "A2",
							                    HOME + OUTPUT_DIR + "C1",
							                    HOME + OUTPUT_DIR + "C2"};
		}
		else {
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
						                        HOME + OUTPUT_DIR + "C"};
		}
		loadTestConfiguration(config);
		
		try 
		{
			//run test cases with single or multiple inputs
			if( type==Type.Multiple )
			{
				long seed1 = System.nanoTime();
				long seed2 = System.nanoTime()+7;
		        
				double[][] A1 = getRandomMatrix(rows, cols, 0, 1, sparsity, seed1);
				double[][] A2 = getRandomMatrix(rows, cols, 0, 1, sparsity, seed2);
		        
				//force binary reblock for 999 to match 1000
		        writeMatrix(A1, HOME + INPUT_DIR + "A1", oi, rows, cols, blocksize-1, blocksize-1);
		        writeMatrix(A2, HOME + INPUT_DIR + "A2", oi, rows, cols, blocksize-1, blocksize-1);
				runTest(true, false, null, -1);
		        double[][] C1 = readMatrix(HOME + OUTPUT_DIR + "C1", InputInfo.BinaryBlockInputInfo, rows, cols, blocksize, blocksize);
		        double[][] C2 = readMatrix(HOME + OUTPUT_DIR + "C2", InputInfo.BinaryBlockInputInfo, rows, cols, blocksize, blocksize);
				
		        TestUtils.compareMatrices(A1, C1, rows, cols, eps);
		        TestUtils.compareMatrices(A2, C2, rows, cols, eps);
			}
			else
			{
				long seed1 = System.nanoTime();
		        double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, seed1);

				//force binary reblock for 999 to match 1000
		        writeMatrix(A, HOME + INPUT_DIR + "A", oi, rows, cols, blocksize-1, blocksize-1);
				runTest(true, false, null, -1);
		        double[][] C = readMatrix(HOME + OUTPUT_DIR + "C", InputInfo.BinaryBlockInputInfo, rows, cols, blocksize, blocksize);
				
		        TestUtils.compareMatrices(A, C, rows, cols, eps);
			}
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			Assert.fail();
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	/**
	 * 
	 * @param ii
	 * @param rows
	 * @param cols
	 * @param brows
	 * @param bcols
	 * @return
	 * @throws IOException 
	 */
	private double[][] readMatrix( String fname, InputInfo ii, long rows, long cols, int brows, int bcols ) 
		throws IOException
	{
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, ii, rows, cols, brows, bcols);
		double[][] C = DataConverter.convertToDoubleMatrix(mb);
		return C;
	}
	
	/**
	 * 
	 * @param A
	 * @param dir
	 * @param oi
	 * @param rows
	 * @param cols
	 * @param brows
	 * @param bcols
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private void writeMatrix( double[][] A, String fname, OutputInfo oi, long rows, long cols, int brows, int bcols ) 
		throws DMLRuntimeException, IOException
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, brows, bcols);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		DataConverter.writeMatrixToHDFS(mb, fname, oi, mc);
		MapReduceTool.writeMetaDataFile(fname+".mtd", ValueType.DOUBLE, mc, oi);
	}
}