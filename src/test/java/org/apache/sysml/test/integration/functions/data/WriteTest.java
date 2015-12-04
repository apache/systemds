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

import static org.junit.Assert.*;

import org.junit.Test;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.test.integration.BinaryMatrixCharacteristics;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>text</li>
 * 	<li>binary</li>
 * 	<li>write a matrix two times</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class WriteTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + WriteTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		
		// positive tests
		addTestConfiguration("TextTest", new TestConfiguration(TEST_CLASS_DIR, "WriteTest", new String[] { "a" }));
		addTestConfiguration("BinaryTest", new TestConfiguration(TEST_CLASS_DIR, "WriteTest", new String[] { "a" }));
		addTestConfiguration("WriteTwiceTest", new TestConfiguration(TEST_CLASS_DIR, "WriteTwiceTest", new String[] { "b", "c" }));
		
		// negative tests
	}
	
	@Test
	public void testText() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("TextTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");	
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrix("a", a);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testBinary() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("BinaryTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
				
		runTest();
		
		//compareResults();
		
		BinaryMatrixCharacteristics matrix = TestUtils.readBlocksFromSequenceFile(output("a"),1000,1000);
		assertEquals(rows, matrix.getRows());
		assertEquals(cols, matrix.getCols());
		double[][] matrixValues = matrix.getValues();
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				assertEquals(i + "," + j, a[i][j], matrixValues[i][j], 0);
			}
		}
	}
	
	@Test
	public void testWriteTwice() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WriteTwiceTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrix("b", a);
		writeExpectedMatrix("c", a);
		
		
		runTest();

			
		compareResults();

	}

}
