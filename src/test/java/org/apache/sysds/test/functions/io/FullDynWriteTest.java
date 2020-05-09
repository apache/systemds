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

package org.apache.sysds.test.functions.io;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FullDynWriteTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME1 = "DynWriteScalar";
	private final static String TEST_NAME2 = "DynWriteMatrix";
	private final static String TEST_DIR = "functions/io/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullDynWriteTest.class.getSimpleName() + "/";
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
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" }) );
	}

	@Test
	public void testScalarCP() 
	{
		runDynamicWriteTest( Type.Scalar, FileFormat.TEXT, ExecType.CP);
	}
	
	@Test
	public void testMatrixTextCP() 
	{
		runDynamicWriteTest( Type.Matrix, FileFormat.TEXT, ExecType.CP);
	}
	
	@Test
	public void testMatrixCSVCP() 
	{
		runDynamicWriteTest( Type.Matrix, FileFormat.CSV, ExecType.CP);
	}
	
	@Test
	public void testMatrixMMCP() 
	{
		runDynamicWriteTest( Type.Matrix, FileFormat.MM, ExecType.CP);
	}
	
	@Test
	public void testMatrixBinaryCP() 
	{
		runDynamicWriteTest( Type.Matrix, FileFormat.BINARY, ExecType.CP);
	}
	
	private void runDynamicWriteTest( Type type, FileFormat fmt, ExecType et )
	{		
		String TEST_NAME = (type==Type.Scalar) ? TEST_NAME1 : TEST_NAME2;
		ExecMode platformOld = rtplatform;
		rtplatform = ExecMode.HYBRID;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{ "-explain","-args",
			input("A"), fmt.toString(), outputDir()};
		
		try 
		{
			long seed1 = System.nanoTime();
			double[][] A = getRandomMatrix(rows, cols, 0, 1, 1.0, seed1);
			writeMatrix(A, input("A"), fmt, rows, cols, 1000, rows*cols);
		
			//run testcase
			runTest(true, false, null, -1);
		
			//check existing file and compare results
			long sum =  computeSum(A);
			String fname = output(Long.toString(sum));
			
			if( type == Type.Scalar ) {
				long val = HDFSTool.readIntegerFromHDFSFile(fname);
				Assert.assertEquals(val, sum);
			}
			else{
				double[][] B = readMatrix(fname, fmt, rows, cols, 1000);
				TestUtils.compareMatrices(A, B, rows, cols, eps);
			}
		
			HDFSTool.deleteFileIfExistOnHDFS(fname);
			HDFSTool.deleteFileIfExistOnHDFS(fname+".mtd");
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
	
	private static double[][] readMatrix( String fname, FileFormat fmt, long rows, long cols, int blen ) 
		throws IOException
	{
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, fmt, rows, cols, blen);
		double[][] C = DataConverter.convertToDoubleMatrix(mb);
		return C;
	}
	
	private static void writeMatrix( double[][] A, String fname, FileFormat fmt, long rows, long cols, int blen, long nnz ) 
		throws IOException
	{
		HDFSTool.deleteFileIfExistOnHDFS(fname);
		HDFSTool.deleteFileIfExistOnHDFS(fname+".mtd");
		
		MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blen, nnz);
		MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
		DataConverter.writeMatrixToHDFS(mb, fname, fmt, mc);
		if( fmt != FileFormat.MM )
			HDFSTool.writeMetaDataFile(fname+".mtd", ValueType.FP64, mc, fmt);
	}
	
	private static long computeSum( double[][] A ) {
		double ret = 0;
		for( int i=0; i<A.length; i++ )
			for( int j=0; j<A[i].length; j++ )
				ret += A[i][j];
		return UtilFunctions.toLong(ret);
	}
}