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

package org.apache.sysml.test.integration.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * The main purpose of this test is to verify the internal optimization regarding
 * sparse-safeness of ppred for various input combinations. (ppred is not sparse-safe 
 * in general, but for certain instance involving 0 scalar it is).
 * 
 * Furthermore, it is used to test all combinations of matrix-scalar, scalar-matrix
 * ppred operations in all execution types.
 * 
 */
public class FullPPredScalarRightTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "PPredScalarRightTest";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullPPredScalarRightTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1072;
	private final static int cols1 = 1009;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS,
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" })   );
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testPPredGreaterZeroDenseCP() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessZeroDenseCP() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsZeroDenseCP() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsZeroDenseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsZeroDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testPPredGreaterNonZeroDenseCP() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessNonZeroDenseCP() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterZeroSparseCP() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessZeroSparseCP() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsZeroSparseCP() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsZeroSparseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsZeroSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testPPredGreaterNonZeroSparseCP() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessNonZeroSparseCP() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testPPredGreaterZeroDenseMR() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessZeroDenseMR() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsZeroDenseMR() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsZeroDenseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsZeroDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.MR);
	}

	@Test
	public void testPPredGreaterNonZeroDenseMR() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessNonZeroDenseMR() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterZeroSparseMR() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessZeroSparseMR() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsZeroSparseMR() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsZeroSparseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsZeroSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.MR);
	}

	@Test
	public void testPPredGreaterNonZeroSparseMR() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessNonZeroSparseMR() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runPPredTest( Type type, boolean zero, boolean sparse, ExecType et )
	{
		String TEST_NAME = TEST_NAME1;
		int rows = rows1;
		int cols = cols1;
		double sparsity = sparse ? sparsity2 : sparsity1;
		double constant = zero ? 0 : 0.5;
		
		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED) {
			TEST_CACHE_DIR = type.ordinal() + "_" + constant + "_" + sparsity + "/";
		}
		
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), 
				Integer.toString(type.ordinal()), Double.toString(constant), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + 
				type.ordinal() + " " + constant + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7); 
			writeInputMatrixWithMTD("A", A, true);
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
}