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

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForAdversarialLiteralsTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1a = "parfor_literals1a"; //local parfor, out filename dynwrite contains _t0
	private final static String TEST_NAME1b = "parfor_literals1b"; //remote parfor, out filename dynwrite contains _t0
	private final static String TEST_NAME1c = "parfor_literals1c"; //local parfor nested, out filename dynwrite contains _t0
	private final static String TEST_NAME2 = "parfor_literals2"; //TODO clarify functions first
	private final static String TEST_NAME3 = "parfor_literals3"; //remote parfor, print delimiters for prog conversion.
	private final static String TEST_NAME4a = "parfor_literals4a"; //local parfor, varname _t0 
	private final static String TEST_NAME4b = "parfor_literals4b"; //remote parfor, varname _t0
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForAdversarialLiteralsTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;
	private final static int cols = 10;    
	private final static double sparsity = 1.0;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1a, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1a, new String[] { "_t0B" }) );
		addTestConfiguration(TEST_NAME1b, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1b, new String[] { "_t0B" }) );
		addTestConfiguration(TEST_NAME1c, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1c, new String[] { "_t0B" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" }) );
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "B" }) );
		addTestConfiguration(TEST_NAME4a, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4a, new String[] { "B" }) );
		addTestConfiguration(TEST_NAME4b, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4b, new String[] { "B" }) );
	}

	@Test
	public void testParForLocalThreadIDLiterals() 
	{
		runLiteralTest(TEST_NAME1a);
	}
	
	@Test
	public void testParForRemoteThreadIDLiterals() 
	{
		runLiteralTest(TEST_NAME1b);
	}
	
	@Test
	public void testParForLocalNestedThreadIDLiterals() 
	{
		runLiteralTest(TEST_NAME1c);
	}
	
	@Test
	public void testParForExtFuncLiterals()  
	{
		runLiteralTest(TEST_NAME2);
	}
	
	@Test
	public void testParForDelimiterLiterals() 
	{
		runLiteralTest(TEST_NAME3);
	}
	
	@Test
	public void testParForLocalThreadIDVarname() 
	{
		runLiteralTest(TEST_NAME4a);
	}
	
	@Test
	public void testParForRemoteThreadIDVarname() 
	{
		runLiteralTest(TEST_NAME4b);
	}

	
	@SuppressWarnings("deprecation")
	private void runLiteralTest( String testName )
	{
		String TEST_NAME = testName;
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		// This is for running the junit test the new way, i.e., construct the arguments directly 
		String HOME = SCRIPT_DIR + TEST_DIR;
		String IN = "A";
		String OUT = (testName.equals(TEST_NAME1a)||testName.equals(TEST_NAME1b))?ProgramConverter.CP_ROOT_THREAD_ID:"B";

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", input(IN),
			Integer.toString(rows), Integer.toString(cols), output(OUT) };
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
		
        double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
		writeInputMatrix("A", A, false);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlin = TestUtils.readDMLMatrixFromHDFS(input(IN));
		HashMap<CellIndex, Double> dmlout = readDMLMatrixFromHDFS(OUT); 
				
		TestUtils.compareMatrices(dmlin, dmlout, eps, "DMLin", "DMLout");			
	}
	
}