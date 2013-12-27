/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.dmlscript;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
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
public class DMLScriptTest2 extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/dmlscript/";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/dmlscript/";

		// positive tests
		
		

		// negative tests
		availableTestConfigurations.put("DMLScriptTest2", new TestConfiguration("functions/dmlscript/", "DMLScriptTest2", new String[] { "a" }));
	}

	@Test
	public void testWithFile() {
		int rows = 10;
		int cols = 10;
		String HOME = SCRIPT_DIR + TEST_DIR;

		TestConfiguration config = availableTestConfigurations.get("DMLScriptTest2");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a, true);
		
		//Expect to print out an ERROR message. -f or -s must be the first argument.
		fullDMLScriptName = baseDirectory + "DMLScriptTest.dml";
		programArgs = new String[]{ "-exec", "hybrid", "-args", HOME + INPUT_DIR + "a" , 
		                       Integer.toString(rows),
		                       Integer.toString(cols),
		                       "text",
		                       HOME + OUTPUT_DIR + "a"
		                       };
		runTest(true, false, null, -1);

		//Expect to print out an ERROR message. -args should be the last argument.
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "a" , 
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a",
	                        "-exec", "hybrid"};
		runTest(true, true, DMLException.class, -1);
		
		//Expect to print out an ERROR message, -de is an unknown argument
		programArgs = new String[]{"-de", "-exec", "hybrid", "-config=" + baseDirectory + "SystemML-config.xml",
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
		
		//Expect to print out an ERROR message, -config syntax is -config=<config file>
		programArgs = new String[]{"-exec", "hybrid", "-config", baseDirectory + "SystemML-config.xml",
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

		TestConfiguration config = availableTestConfigurations.get("DMLScriptTest2");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");
		
		loadTestConfiguration(config);

		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.5, -1);
		writeInputMatrix("a", a, true);

		
		//Expect to print out an ERROR message. -f or -s must be the first argument.
		programArgs = new String[]{ "-v", "-s", s, 
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
	
		
		runTest(true, false, null, -1);
		
		
		//Expect to print out an ERROR message. -args should be the last argument.
		programArgs = new String[]{"-s", s, 
	               "-args", "-v", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
	
		
		runTest(true, false, null, -1);
		
		//Expect to print out an ERROR message, -de is an unknown argument
		programArgs = new String[]{"-s", s, "-de",
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
		
		//Expect to print out an ERROR message, -config syntax is -config=<config file>
		programArgs = new String[]{"-s", s, "-config", baseDirectory + "SystemML-config.xml", "-exec", "hybrid", 
	               "-args", HOME + INPUT_DIR + "a" ,
	                        Integer.toString(rows),
	                        Integer.toString(cols),
	                        "text",
	                        HOME + OUTPUT_DIR + "a"};
		runTest(true, false, null, -1);
	}
}
