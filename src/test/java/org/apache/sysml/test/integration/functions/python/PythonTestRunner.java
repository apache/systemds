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
package org.apache.sysml.test.integration.functions.python;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.ProcessBuilder.Redirect;
import java.util.Map;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Test;

/**
 * To run Python tests, please:
 * 1. Set the RUN_PYTHON_TEST flag to true.
 * 2. Set SPARK_HOME environment variable
 * 3. Compile SystemML so that there is SystemML.jar in the target directory
 */
public class PythonTestRunner extends AutomatedTestBase
{
	
	private static boolean RUN_PYTHON_TEST = false;
	
	private final static String TEST_NAME = "PythonTestRunner";
	private final static String TEST_DIR = "functions/python/";
	private final static String TEST_CLASS_DIR = TEST_DIR + PythonTestRunner.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"B"}));
	}
	
	
	@Test
	public void testMLContext() throws DMLRuntimeException, IOException, InterruptedException  {
		runPythonTest("test_mlcontext.py");
	}
	
	@Test
	public void testMatrixBinaryOp() throws DMLRuntimeException, IOException, InterruptedException  {
		runPythonTest("test_matrix_binary_op.py");
	}
	
	@Test
	public void testMatrixAggFn() throws DMLRuntimeException, IOException, InterruptedException  {
		runPythonTest("test_matrix_agg_fn.py");
	}
	
	@Test
	public void testMLLearn_df() throws DMLRuntimeException, IOException, InterruptedException  {
		runPythonTest("test_mllearn_df.py");
	}
	
	@Test
	public void testMLLearn_numpy() throws DMLRuntimeException, IOException, InterruptedException  {
		runPythonTest("test_mllearn_numpy.py");
	}
	
	public void runPythonTest(String pythonFileName) throws IOException, DMLRuntimeException, InterruptedException {
		if(!RUN_PYTHON_TEST)
			return;
			
		if(!new File("target/SystemML.jar").exists()) {
			throw new DMLRuntimeException("Please build the project before running PythonTestRunner");
		}
//		String [] args = { "--master", "local[*]", "--driver-class-path", "target/SystemML.jar", "src/main/python/tests/test_mlcontext.py"};
//		org.apache.spark.deploy.SparkSubmit.main(args);
		Map<String, String> env = System.getenv();
		if(!env.containsKey("SPARK_HOME")) {
			throw new DMLRuntimeException("Please set the SPARK_HOME environment variable");
		}
		String spark_submit = env.get("SPARK_HOME") + File.separator + "bin" + File.separator + "spark-submit";
		if (System.getProperty("os.name").contains("Windows")) {
			spark_submit += ".cmd";
		}
		Process p = new ProcessBuilder(spark_submit, "--master", "local[*]", 
				"--driver-class-path", "target/SystemML.jar", "src/main/python/tests/" + pythonFileName)
				.redirectError(Redirect.INHERIT)
				.start();
		
		BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
	    String line;
	    boolean passed = false;
	    while ((line = in.readLine()) != null) {
	    	if(line.trim().equals("OK")) {
	    		passed = true;
	    	}
	    	System.out.println(line);
	    }
	    
		// System.out.println( IOUtils.toString(p.getInputStream(), Charset.defaultCharset())); 

		p.waitFor();
		
		if(!passed) {
			throw new DMLRuntimeException("The python test failed:" + pythonFileName);
		}
	}
}
