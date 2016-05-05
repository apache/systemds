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

package org.apache.sysml.test.integration.functions.misc;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.utils.Statistics;

public class FunctionNamespaceTest extends AutomatedTestBase 
{
	private final static String TEST_NAME0 = "FunctionsA";
	private final static String TEST_NAME1 = "Functions1";
	private final static String TEST_NAME2 = "Functions2";
	private final static String TEST_NAME3 = "Functions3";
	private final static String TEST_NAME4 = "Functions4";
	private final static String TEST_NAME5 = "Functions5";
	private final static String TEST_NAME6 = "Functions6";
	private final static String TEST_NAME7 = "Functions7";
	private final static String TEST_NAME8 = "Functions8";
	private final static String TEST_NAME9 = "Functions9";
	private final static String TEST_NAME10 = "Functions10";
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionNamespaceTest.class.getSimpleName() + "/";
	
	private final static long rows = 3400;
	private final static long cols = 2700;
	private final static double val = 1.0;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME0, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME0));
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1)); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2)); 
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3)); 
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4)); 
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5)); 
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6)); 
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7)); 
		addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8)); 
		addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9)); 
		addTestConfiguration(TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10)); 
	}
	
	@Test
	public void testFunctionDefaultNS() 
	{
		runFunctionNamespaceTest(TEST_NAME0, ScriptType.DML);
	}
	
	@Test
	public void testFunctionSourceNS() 
	{
		runFunctionNamespaceTest(TEST_NAME1, ScriptType.DML);
	}
	
	@Test
	public void testFunctionWithoutNS() 
	{
		runFunctionNamespaceTest(TEST_NAME2, ScriptType.DML);
	}
	
	@Test
	public void testFunctionImportSource() 
	{
		runFunctionNamespaceTest(TEST_NAME3, ScriptType.DML);
	}
	
	@Test
	public void testFunctionMultiSource() 
	{
		runFunctionNamespaceTest(TEST_NAME4, ScriptType.DML);
	}
	
	@Test
	public void testFunctionNoInliningIPA() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME5, ScriptType.DML, true);
	}
	
	@Test
	public void testFunctionNoInliningNoIPA() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME5, ScriptType.DML, false);
	}
	
	@Test
	public void testFunctionCircular() 
	{
		runFunctionNamespaceTest(TEST_NAME6, ScriptType.DML);
	}
	
	@Test
	public void testFunctionCircularChain() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME7, ScriptType.DML, true);
	}
	
	@Test
	public void testFunctionCircularChainNoIPA() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME7, ScriptType.DML, false);
	}
	@Test
	public void testFunctionErrorConflict() 
	{
		runFunctionNamespaceTest(TEST_NAME8, ScriptType.DML);
	}
	
	@Test
	public void testFunctionIndirectConflict() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME9, ScriptType.DML, true);
	}
	
	@Test
	public void testFunctionMultiConflict() 
	{
		runFunctionNamespaceTest(TEST_NAME10, ScriptType.DML);
	}
	
	@Test
	public void testPyFunctionDefaultNS() 
	{
		runFunctionNamespaceTest(TEST_NAME0, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionSourceNS() 
	{
		runFunctionNamespaceTest(TEST_NAME1, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionWithoutNS() 
	{
		runFunctionNamespaceTest(TEST_NAME2, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionImportSource() 
	{
		runFunctionNamespaceTest(TEST_NAME3, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionMultiSource() 
	{
		runFunctionNamespaceTest(TEST_NAME4, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionNoInliningIPA() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME5, ScriptType.PYDML, true);
	}
	
	@Test
	public void testPyFunctionNoInliningNoIPA() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME5, ScriptType.PYDML, false);
	}
	
	@Test
	public void testPyFunctionCircular() 
	{
		runFunctionNamespaceTest(TEST_NAME6, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionCircularChain() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME7, ScriptType.PYDML, true);
	}
	
	@Test
	public void testPyFunctionCircularChainNoIPA() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME7, ScriptType.PYDML, false);
	}
	
	@Test
	public void testPyFunctionErrorConflict() 
	{
		runFunctionNamespaceTest(TEST_NAME8, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionIndirectConflict() 
	{
		runFunctionNoInliningNamespaceTest(TEST_NAME9, ScriptType.PYDML, true);
	}
	
	@Test
	public void testPyFunctionMultiConflict() 
	{
		runFunctionNamespaceTest(TEST_NAME10, ScriptType.PYDML);
	}

	private void runFunctionNamespaceTest(String TEST_NAME, ScriptType scriptType)
	{		
		getAndLoadTestConfiguration(TEST_NAME);
		
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "." + scriptType.toString().toLowerCase();
		programArgs = (ScriptType.PYDML == scriptType) ? new String[]{"-python"} : new String[]{};
		
		PrintStream origStdErr = System.err;

		try
		{
			ByteArrayOutputStream baos = null;
			
			boolean exceptionExpected = (TEST_NAME2.equals(TEST_NAME)) ? true : false;
			if (!exceptionExpected)
			{
				baos = new ByteArrayOutputStream();
				PrintStream newStdErr = new PrintStream(baos);
				System.setErr(newStdErr);
			}
			
			runTest(true, exceptionExpected, DMLException.class, -1);
			
			if (!exceptionExpected)
			{
				String stdErrString = baos.toString();
				if (null != stdErrString && stdErrString.length() > 0)
				{
					if (TEST_NAME8.equals(TEST_NAME))
					{
						if (!stdErrString.contains("Namespace Conflict"))
						{
							Assert.fail("Expected parse issue not detected.");
						}
					}
					else
					{
						Assert.fail("Unexpected parse error or DML script error: " + stdErrString);
					}
				}
			}
		}
		catch (Exception e) 
		{
			e.printStackTrace(origStdErr);
			Assert.fail("Unexpected exception: " + e);
		}
		finally
		{
			System.setErr(origStdErr);
		}
	}

	/*
	 * Based on testChainNoInliningIPA and testChainNoInliningIPA from
	 * org.apache.sysml.test.integration.functions.misc.FunctionInliningTest
	 */
	private void runFunctionNoInliningNamespaceTest(String TEST_NAME, ScriptType scriptType, boolean IPA)
	{		
		boolean origIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		getAndLoadTestConfiguration(TEST_NAME);
		
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "." + scriptType.toString().toLowerCase();
		programArgs = (ScriptType.PYDML == scriptType) ? 
			new String[]{"-python", "-args", String.valueOf(rows), String.valueOf(cols), String.valueOf(val), output("Rout")} : 
			new String[]{"-args", String.valueOf(rows), String.valueOf(cols), String.valueOf(val), output("Rout")};
		
		PrintStream originalStdErr = System.err;

		try
		{
			ByteArrayOutputStream baos = new ByteArrayOutputStream();
			PrintStream newStdErr = new PrintStream(baos);
			System.setErr(newStdErr);
	        
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			runTest(true, false, null, -1); 
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = origIPA;
			
			//compare output
			double ret = MapReduceTool.readDoubleFromHDFSFile(output("Rout"));
			Assert.assertEquals(Double.valueOf(rows*cols*val*6), Double.valueOf(ret));
			
			//compiled MR jobs
			int expectNumCompiled = IPA ? 0 : 4; 
			Assert.assertEquals("Unexpected number of compiled MR jobs.", expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//check executed MR jobs (should always be 0 due to dynamic recompilation)
			int expectNumExecuted = 0;
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectNumExecuted, Statistics.getNoOfExecutedMRJobs());
			
			String stdErrString = baos.toString();
			if (stdErrString != null && stdErrString.length() > 0)
			{
				Assert.fail("Unexpected parse error or DML script error: " + stdErrString);
			}
		}
		catch (Exception e) 
		{
			e.printStackTrace(originalStdErr);
			Assert.fail("Unexpected exception: " + e);
		}
		finally
		{
			System.setErr(originalStdErr);
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = origIPA;
		}
	}
}