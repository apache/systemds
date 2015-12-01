/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.data;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

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
	public void testTextCellSingleMDenseSP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellSingeMSparseSP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellSingleVDenseSP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellSingeVSparseSP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellMultipleMDenseSP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, false, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testTextCellMultipleMSparseSP() 
	{
		runReblockTest(OutputInfo.TextCellOutputInfo, true, Type.Multiple, ExecType.SPARK);
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

	@Test
	public void testBinaryBlockSingleMDenseSP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingeMSparseSP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingleVDenseSP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockSingeVSparseSP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockMultipleMDenseSP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, false, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testBinaryBlockMultipleMSparseSP() 
	{
		runReblockTest(OutputInfo.BinaryBlockOutputInfo, true, Type.Multiple, ExecType.SPARK);
	}

	//csv
	
	@Test
	public void testCSVSingleMDenseCP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testCSVSingeMSparseCP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Single, ExecType.CP);
	}
	
	@Test
	public void testCSVSingleVDenseCP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testCSVSingeVSparseCP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Vector, ExecType.CP);
	}
	
	@Test
	public void testCSVMultipleMDenseCP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testCSVMultipleMSparseCP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Multiple, ExecType.CP);
	}
	
	@Test
	public void testCSVSingleMDenseSP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testCSVSingeMSparseSP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Single, ExecType.SPARK);
	}
	
	@Test
	public void testCSVSingleVDenseSP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testCSVSingeVSparseSP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Vector, ExecType.SPARK);
	}
	
	@Test
	public void testCSVMultipleMDenseSP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testCSVMultipleMSparseSP() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Multiple, ExecType.SPARK);
	}
	
	@Test
	public void testCSVSingleMDenseMR() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Single, ExecType.MR);
	}
	
	@Test
	public void testCSVSingeMSparseMR() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Single, ExecType.MR);
	}
	
	
	@Test
	public void testCSVSingleVDenseMR() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Vector, ExecType.MR);
	}
	
	@Test
	public void testCSVSingeVSparseMR() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Vector, ExecType.MR);
	}
	
	@Test
	public void testCSVMultipleMDenseMR() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, false, Type.Multiple, ExecType.MR);
	}
	
	@Test
	public void testCSVMultipleMSparseMR() 
	{
		runReblockTest(OutputInfo.CSVOutputInfo, true, Type.Multiple, ExecType.MR);
	}

	/**
	 * 
	 * @param oi
	 * @param sparse
	 * @param type
	 * @param et
	 */
	private void runReblockTest( OutputInfo oi, boolean sparse, Type type, ExecType et )
	{		
		String TEST_NAME = (type==Type.Multiple) ? TEST_NAME2 : TEST_NAME1;		 
		double sparsity = (sparse) ? sparsity2 : sparsity1;		
		int rows = (type==Type.Vector)? rowsV : rowsM;
		int cols = (type==Type.Vector)? colsV : colsM;
		
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
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
		        writeMatrix(A1, input("A1"), oi, rows, cols, blocksize-1, blocksize-1);
		        writeMatrix(A2, input("A2"), oi, rows, cols, blocksize-1, blocksize-1);
				runTest(true, false, null, -1);
		        double[][] C1 = readMatrix(output("C1"), InputInfo.BinaryBlockInputInfo, rows, cols, blocksize, blocksize);
		        double[][] C2 = readMatrix(output("C2"), InputInfo.BinaryBlockInputInfo, rows, cols, blocksize, blocksize);
				
		        TestUtils.compareMatrices(A1, C1, rows, cols, eps);
		        TestUtils.compareMatrices(A2, C2, rows, cols, eps);
			}
			else
			{
				long seed1 = System.nanoTime();
		        double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, seed1);

				//force binary reblock for 999 to match 1000
		        writeMatrix(A, input("A"), oi, rows, cols, blocksize-1, blocksize-1);
				runTest(true, false, null, -1);
		        double[][] C = readMatrix(output("C"), InputInfo.BinaryBlockInputInfo, rows, cols, blocksize, blocksize);
				
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
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
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