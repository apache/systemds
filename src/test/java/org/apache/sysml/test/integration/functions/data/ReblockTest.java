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

package org.apache.sysml.test.integration.functions.data;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>decrease block size</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class ReblockTest extends AutomatedTestBase 
{

	private static final String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ReblockTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		
		// positive tests
		addTestConfiguration("ReblockTest", 
			new TestConfiguration(TEST_CLASS_DIR, "ReblockTest", new String[] { "a" }));
		
		// negative tests
	}
	
	@Test
	public void testReblock() {
		TestConfiguration config = getTestConfiguration("ReblockTest");
		loadTestConfiguration(config);
		
		int rows = 10;
		int cols = 10;

		double[][] a = getRandomMatrix(rows, cols, 1, 1, 1, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrix("a", a);
		
		runTest();
		
		compareResults();
	}

}
