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

package org.apache.sysml.test.integration.functions.binary.matrix;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.utils.Statistics;

/**
 * Tests the number of mapmult operations that can be piggybacked into the same GMR job.
 *
 */

public class MapMultLimitTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "MapMultLimitTest";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + MapMultLimitTest.class.getSimpleName() + "/";
	
	private final static int rows1 = 2000;
	private final static int rows2 = 3500;
	private final static int cols = 1500;
	
	private final static double sparsity1 = 0.7;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "C1", "C2" }) ); 
	}
	
	@Test
	public void testMapMultLimit()
	{

		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = RUNTIME_PLATFORM.HADOOP;

		try
		{
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", 
				input("A"), input("B1"), input("B2"), output("C1"), output("C2") };
	
			//System.out.println("Generating A ...");
			double[][] A = getRandomMatrix(rows1, rows2, 0, 1, sparsity1, 10); 
			MatrixCharacteristics mc = new MatrixCharacteristics(rows1, rows2, -1, -1, -1);
			writeInputMatrixWithMTD("A", A, true, mc);			
			
			//System.out.println("Generating B1 ...");
			double[][] B1 = getRandomMatrix(rows2, cols, 0, 1, 0.4, 10); 
			mc = new MatrixCharacteristics(rows2, cols, -1, -1, -1);
			writeInputMatrixWithMTD("B1", B1, true, mc);

			//System.out.println("Generating B2 ...");
			double[][] B2 = getRandomMatrix(rows2, cols, 0, 1, 0.4, 20); 
			mc = new MatrixCharacteristics(rows2, cols, -1, -1, -1);
			writeInputMatrixWithMTD("B2", B2, true, mc);
	
			//System.out.println("Running test...");
			boolean exceptionExpected = false;
			
			// Expected 3 jobs: 1 Reblock, 2 MapMults
			runTest(true, exceptionExpected, null, 3); 
			//System.out.println("#Jobs: " + Statistics.getNoOfExecutedMRJobs() + ", " + Statistics.getNoOfCompiledMRJobs());
			Assert.assertTrue(Statistics.getNoOfExecutedMRJobs()==3);
		}
		finally
		{
			rtplatform = rtold;
		}
	}
	
}