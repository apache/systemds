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

package org.apache.sysml.test.integration.functions.dmlscript;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;


/**
 * <p>
 * <b>Positive tests:</b>
 * </p>
 * <ul>
 * <li>text format</li>
 * <li>binary format</li>
 * </ul>
 * <p>
 * <b>Negative tests:</b>
 * </p>
 * <ul>
 * </ul>
 * 
 * 
 */
public class DMLScriptTest1 extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/dmlscript/";
	private final static String TEST_CLASS_DIR = TEST_DIR + DMLScriptTest1.class.getSimpleName() + "/";
	private final static String TEST_NAME = "DMLScriptTest";
	
	@Override
	public void setUp() {
		// positive tests
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "a" });
		addTestConfiguration(TEST_NAME, config);
		
		// negative tests		
	}

	@Test
	public void testWithFile() {
		int rows = 10;
		int cols = 10;

		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "DMLScriptTest.dml";
		
		programArgs = new String[]{"-args", input("a"),
			Integer.toString(rows), Integer.toString(cols), "text", output("a")};

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a, true);

		runTest(true, false, null, -1);

		programArgs = new String[]{"-args", input("a"),
			Integer.toString(rows), Integer.toString(cols), "text", output("a")};
		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-exec", "hybrid", "-args", input("a"),
			Integer.toString(rows), Integer.toString(cols), "text", output("a")};
		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-exec", "hybrid", "-config=" + HOME + "SystemML-config.xml",
			"-args", input("a"), Integer.toString(rows), Integer.toString(cols), "text", output("a")};
		runTest(true, false, null, -1);
	}

	@Test
	public void testWithString() {
		String s = " A = read($1, rows=$2, cols=$3, format=$4); \n " + 
				  "write(A, $5, format=$4); \n";
		int rows = 10;
		int cols = 10;
		String HOME = SCRIPT_DIR + TEST_DIR;

		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		loadTestConfiguration(config);
		
		programArgs = new String[]{"-s", s, "-args", input("a"),
			Integer.toString(rows), Integer.toString(cols), "text", output("a")};

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a, true);

		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-s", s, "-args", input("a"),
			Integer.toString(rows), Integer.toString(cols), "text", output("a")};
		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-s", s, "-config=" + HOME + "SystemML-config.xml", "-exec", "hybrid",
			"-args", input("a"), Integer.toString(rows), Integer.toString(cols), "text", output("a")};
		runTest(true, false, null, -1);
	}
}
