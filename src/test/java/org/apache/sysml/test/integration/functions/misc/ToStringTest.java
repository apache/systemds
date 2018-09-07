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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

public class ToStringTest extends AutomatedTestBase {

	 private static final String TEST_DIR = "functions/misc/";
	 private static final String TEST_CLASS_DIR = TEST_DIR + ToStringTest.class.getSimpleName() + "/";
	 private static final String OUTPUT_NAME = "tostring";

	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}
	
	/**
	 * Default parameters
	 */
	@Test
	public void testdefaultPrint(){
		String testName = "ToString1";
		String expectedOutput = 
				"1.000 2.000 3.000 4.000 5.000\n" +  
				"6.000 7.000 8.000 9.000 10.000\n"+
				"11.000 12.000 13.000 14.000 15.000\n" +
				"16.000 17.000 18.000 19.000 20.000\n";
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	
	/**
	 * Specify number of rows and columns on small matrix
	 */
	@Test
	public void testRowsColsPrint(){
		String testName = "ToString2";
		String expectedOutput = 
				"1.000 2.000 3.000\n" +
				"5.000 6.000 7.000\n" +
				"9.000 10.000 11.000\n";		
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}

	/**
	 * Change number of digits after decimal
	 */
	@Test
	public void testDecimal(){
		String testName = "ToString3";
		String expectedOutput = 
				"1.00 2.00 3.00\n" +
				"4.00 5.00 6.00\n" +
				"7.00 8.00 9.00\n";		
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	
	/**
	 * Change separator character
	 */
	@Test
	public void testSeparator(){
		String testName = "ToString4";
		String expectedOutput = 
				"1.000 | 2.000 | 3.000\n" +
				"4.000 | 5.000 | 6.000\n" +
				"7.000 | 8.000 | 9.000\n";		
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	
	/**
	 * Change line separator char
	 */
	@Test
	public void testLineSeparator(){
		String testName = "ToString5";
		String expectedOutput = 
				"1.000 2.000 3.000\t" +
				"4.000 5.000 6.000\t" +
				"7.000 8.000 9.000\t";		
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	

	/**
	 * Initialize a big matrix ( > 100x100), print to see if defaults kick in
	 */
	@Test
	public void testBiggerArrayDefaultRowsCols(){
		//script-consistency required: 200x200
		//TODO pass as arguments
		final int INPUT_COLS = 200;
		final int MAX_ROWS = 100;
		final int MAX_COLS = 100;
		final String SEP = " ";
		final String LINESEP = "\n";
				
		String testName = "ToString6";
		StringBuilder sb = new StringBuilder();
		long k=1;
		long i=1, j=1;
		for (i=1; i<=MAX_ROWS; i++){
			for (j=1; j<=MAX_COLS-1; j++){
				sb.append(k).append(".000").append(SEP);
				k++;
			}
			sb.append(k).append(".000").append(LINESEP);
			k++; j++;
			k += (INPUT_COLS - j + 1);
		}
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, sb.toString());
	}
	
	/**
	 * Initialize a big matrix ( > 100x100), specify rows and cols, bigger than default
	 */
	@Test
	public void testBiggerArraySpecifyRowsCols(){
		//script-consistency required: 200x200
		//TODO pass as arguments
		final int INPUT_COLS = 200;
		final int MAX_ROWS = 190;
		final int MAX_COLS = 190;
		final String SEP = " ";
		final String LINESEP = "\n";
				
		String testName = "ToString7";
		StringBuilder sb = new StringBuilder();
		long k=1;
		long i=1, j=1;
		for (i=1; i<=MAX_ROWS; i++){
			for (j=1; j<=MAX_COLS-1; j++){
				sb.append(k).append(".000").append(SEP);
				k++;
			}
			sb.append(k).append(".000").append(LINESEP);
			k++; j++;
			k += (INPUT_COLS - j + 1);
		}
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, sb.toString());
	}
	
	/**
	 * Basic sparse print test
	 */
	@Test
	public void testSparsePrint(){
		String testName = "ToString8";
		String expectedOutput = "1 1 1.000\n" +
								"1 2 2.000\n" +
								"2 1 3.000\n" +
								"2 2 4.000\n" +
								"3 1 5.000\n" +
								"3 2 6.000\n" +
								"4 1 7.000\n" +
								"4 2 8.000\n";
								
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	
	/**
	 * Basic sparse print test with zeroes
	 */
	@Test
	public void testSparsePrintWithZeroes(){
		String testName = "ToString9";
		String expectedOutput = "1 1 1.000\n" +
								"1 2 2.000\n" +
								"1 3 3.000\n" +
								"1 4 4.000\n" +
								"3 1 5.000\n" +
								"3 2 6.000\n" +
								"3 3 7.000\n";
								
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	
	/**
	 * Sparse print with specified separator and lineseparator
	 */
	@Test
	public void testSparsePrintWithZeroesAndFormatting(){
		String testName = "ToString10";
		String expectedOutput = "1  1  1.000|" +
								"1  2  2.000|" +
								"1  3  3.000|" +
								"1  4  4.000|" +
								"3  1  5.000|" +
								"3  2  6.000|" +
								"3  3  7.000|";
								
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}
	
	/**
	 * Sparse print with custom number of rows and columns
	 */
	@Test
	public void testSparsePrintWithZeroesRowsCols(){
		String testName = "ToString11";
		String expectedOutput = "1 1 1.000\n" +
								"1 2 2.000\n" +
								"1 3 3.000\n" +
								"2 1 2.000\n" +
								"3 1 5.000\n" +
								"3 2 6.000\n" +
								"3 3 7.000\n";
								
		addTestConfiguration(testName, new TestConfiguration(TEST_CLASS_DIR, testName));
		toStringTestHelper(RUNTIME_PLATFORM.SINGLE_NODE, testName, expectedOutput);
	}

	protected void toStringTestHelper(RUNTIME_PLATFORM platform, String testName, String expectedOutput) {
		RUNTIME_PLATFORM platformOld = rtplatform;
		
		rtplatform = platform;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
        if (rtplatform == RUNTIME_PLATFORM.SPARK)
            DMLScript.USE_LOCAL_SPARK_CONFIG = true;
        try {
            // Create and load test configuration
        	getAndLoadTestConfiguration(testName);
            String HOME = SCRIPT_DIR + TEST_DIR;
            fullDMLScriptName = HOME + testName + ".dml";
            programArgs = new String[]{"-args", output(OUTPUT_NAME)};


            // Run DML and R scripts
            runTest(true, false, null, -1);

            // Compare output strings
            String output = TestUtils.readDMLString(output(OUTPUT_NAME));
            TestUtils.compareScalars(expectedOutput, output);
           
        }
        finally {
            // Reset settings
            rtplatform = platformOld;
            DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
        }
	}
	

}
