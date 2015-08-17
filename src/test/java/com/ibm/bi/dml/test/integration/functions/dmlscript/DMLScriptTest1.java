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

package com.ibm.bi.dml.test.integration.functions.dmlscript;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


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
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/dmlscript/";

		// positive tests
		availableTestConfigurations.put("DMLScriptTest", new TestConfiguration("functions/dmlscript/", "DMLScriptTest", new String[] { "a" }));
		

		// negative tests
		
	}

	@Test
	public void testWithFile() {
		int rows = 10;
		int cols = 10;
		String HOME = SCRIPT_DIR + TEST_DIR;

		TestConfiguration config = availableTestConfigurations.get("DMLScriptTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		fullDMLScriptName = baseDirectory + "DMLScriptTest.dml";
		
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
	
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a, true);

		runTest(true, false, null, -1);

		programArgs = new String[]{"-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
		
		
		programArgs = new String[]{"-exec", "hybrid",
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-exec", "hybrid", "-config=" + baseDirectory + "SystemML-config.xml",
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
		
	}

	@Test
	public void testWithString() {
		String s = " A = read($1, rows=$2, cols=$3, format=$4); \n " + 
				  "write(A, $5, format=$4); \n";
		int rows = 10;
		int cols = 10;
		String HOME = SCRIPT_DIR + TEST_DIR;

		TestConfiguration config = availableTestConfigurations.get("DMLScriptTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		programArgs = new String[]{"-s", s, 
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
	
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a, true);

		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-s", s,
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
		
		programArgs = new String[]{"-s", s, "-config=" + baseDirectory + "SystemML-config.xml", "-exec", "hybrid", 
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
	}
}
