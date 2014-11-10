/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io;

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
import com.ibm.bi.dml.runtime.util.UtilFunctions;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullDynWriteTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "DynWriteScalar";
	private final static String TEST_NAME2 = "DynWriteMatrix";
	private final static String TEST_DIR = "functions/io/";
	private final static double eps = 1e-10;
	
	private final static int rows = 350;
	private final static int cols = 110; 
	
	public enum Type{
		Scalar,
		Matrix
	} 
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "B" })   );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "B" })   );
	}

	@Test
	public void testScalarCP() 
	{
		runDynamicWriteTest( Type.Scalar, OutputInfo.TextCellOutputInfo, ExecType.CP);
	}
	
	@Test
	public void testMatrixTextCP() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.TextCellOutputInfo, ExecType.CP);
	}
	
	@Test
	public void testMatrixCSVCP() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.CSVOutputInfo, ExecType.CP);
	}
	
	@Test
	public void testMatrixMMCP() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.MatrixMarketOutputInfo, ExecType.CP);
	}
	
	@Test
	public void testMatrixBinaryCP() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.BinaryBlockOutputInfo, ExecType.CP);
	}
	
	@Test
	public void testScalarMR() 
	{
		runDynamicWriteTest( Type.Scalar, OutputInfo.TextCellOutputInfo, ExecType.MR);
	}
	
	@Test
	public void testMatrixTextMR() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.TextCellOutputInfo, ExecType.MR);
	}
	
	@Test
	public void testMatrixCSVMR() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.CSVOutputInfo, ExecType.MR);
	}
	
	@Test
	public void testMatrixMMMR() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.MatrixMarketOutputInfo, ExecType.MR);
	}
	
	@Test
	public void testMatrixBinaryMR() 
	{
		runDynamicWriteTest( Type.Matrix, OutputInfo.BinaryBlockOutputInfo, ExecType.MR);
	}
	
	/**
	 * 
	 * @param type
	 * @param fmt
	 * @param et
	 */
	private void runDynamicWriteTest( Type type, OutputInfo fmt, ExecType et )
	{		
		String TEST_NAME = (type==Type.Scalar) ? TEST_NAME1 : TEST_NAME2;		 
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{ "-explain","-args", HOME + INPUT_DIR + "A",
				                           getFormatString(fmt),
				                           HOME + OUTPUT_DIR};
		loadTestConfiguration(config);
		
		try 
		{		
			long seed1 = System.nanoTime();
		    double[][] A = getRandomMatrix(rows, cols, 0, 1, 1.0, seed1);
		    writeMatrix(A, HOME + INPUT_DIR + "A", fmt, rows, cols, 1000, 1000, rows*cols);
		    
		    //run testcase
			runTest(true, false, null, -1);
		    
			//check existing file and compare results
			long sum =  computeSum(A);
			String fname = HOME + OUTPUT_DIR + sum;
			
			if( type == Type.Scalar ) {
				long val = MapReduceTool.readIntegerFromHDFSFile(fname);
				Assert.assertEquals(val, sum);
			}
			else{
				double[][] B = readMatrix(fname, fmt.getMatchingInputInfo(fmt), rows, cols, 1000, 1000);
			    TestUtils.compareMatrices(A, B, rows, cols, eps);
			}
		    
		    MapReduceTool.deleteFileIfExistOnHDFS(fname);
		    MapReduceTool.deleteFileIfExistOnHDFS(fname+".mtd");
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
	private void writeMatrix( double[][] A, String fname, OutputInfo oi, long rows, long cols, int brows, int bcols, long nnz ) 
		throws DMLRuntimeException, IOException
	{
		MapReduceTool.deleteFileIfExistOnHDFS(fname);
		MapReduceTool.deleteFileIfExistOnHDFS(fname+".mtd");
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, brows, bcols, nnz);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		DataConverter.writeMatrixToHDFS(mb, fname, oi, mc);
		if( oi != OutputInfo.MatrixMarketOutputInfo )
			MapReduceTool.writeMetaDataFile(fname+".mtd", ValueType.DOUBLE, mc, oi);
	}
	
	/**
	 * 
	 * @param oinfo
	 * @return
	 */
	private String getFormatString(OutputInfo oinfo)
	{
		if( oinfo==OutputInfo.BinaryBlockOutputInfo )
			return "binary";
		else if( oinfo==OutputInfo.TextCellOutputInfo )
			return "text";
		else if( oinfo==OutputInfo.MatrixMarketOutputInfo )
			return "mm";
		else if( oinfo==OutputInfo.CSVOutputInfo )
			return "csv";
		
		return null;
	}
	
	/**
	 * 
	 * @param A
	 * @return
	 */
	private long computeSum( double[][] A )
	{
		double ret = 0;
		
		for( int i=0; i<A.length; i++ )
			for( int j=0; j<A[i].length; j++ )
				ret += A[i][j];
		
		return UtilFunctions.toLong(ret);
	}
	
}