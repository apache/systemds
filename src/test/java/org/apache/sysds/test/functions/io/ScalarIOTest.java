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

package org.apache.sysds.test.functions.io;

import java.io.ByteArrayOutputStream;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class ScalarIOTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "scalarIOTest";
	private final static String TEST_DIR = "functions/io/";
	private final static String OUT_FILE = "a.scalar";
	private final static String TEST_CLASS_DIR = TEST_DIR + ScalarIOTest.class.getSimpleName() + "/";
	private final static String HOME = SCRIPT_DIR + TEST_DIR;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "a.scalar" }) );
		
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testIntScalarWrite() {

		int int_scalar = 464;
		
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", String.valueOf(int_scalar), output("a.scalar") };
		runTest(true, false, null, -1);
		
		int int_out_scalar = TestUtils.readDMLScalarFromHDFS(output(OUT_FILE)).get(new CellIndex(1,1)).intValue();
		Assert.assertEquals("Values not equal: " + int_scalar + Opcodes.NOTEQUAL.toString() + int_out_scalar, int_scalar, int_out_scalar);
		
		// Invoke the DML script that does computations and then writes scalar to HDFS
		fullDMLScriptName = HOME + "ScalarComputeWrite.dml";
		runTest(true, false, null, -1);
		
		int_out_scalar = TestUtils.readDMLScalarFromHDFS(output(OUT_FILE)).get(new CellIndex(1,1)).intValue();
		Assert.assertEquals("Computation test for Integers failed: Values not equal: " + int_scalar + Opcodes.NOTEQUAL.toString() + int_out_scalar, int_scalar, int_out_scalar);
	}

	@Test
	public void testDoubleScalarWrite() 
	{
		Double double_scalar = 464.55;

		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", String.valueOf(double_scalar), output("a.scalar") };
		runTest(true, false, null, -1);
		
		Double double_out_scalar = TestUtils.readDMLScalarFromHDFS(output(OUT_FILE)).get(new CellIndex(1,1)).doubleValue();
		Assert.assertEquals("Values not equal: " + double_scalar + Opcodes.NOTEQUAL.toString() + double_out_scalar, double_scalar, double_out_scalar);

		// Invoke the DML script that does computations and then writes scalar to HDFS
		fullDMLScriptName = HOME + "ScalarComputeWrite.dml";
		runTest(true, false, null, -1);
		
		double_out_scalar = TestUtils.readDMLScalarFromHDFS(output(OUT_FILE)).get(new CellIndex(1,1)).doubleValue();
		Assert.assertEquals("Computation test for Integers failed: Values not equal: " + double_scalar + Opcodes.NOTEQUAL.toString() + double_out_scalar, double_scalar, double_out_scalar);
	}

	@Test
	public void testBooleanScalarWrite() {

		boolean boolean_scalar = true;

		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", String.valueOf(boolean_scalar), output("a.scalar") };
		runTest(true, false, null, -1);

		boolean boolean_out_scalar = TestUtils.readDMLBoolean(output(OUT_FILE));
		
		Assert.assertEquals("Values not equal: " + boolean_scalar + Opcodes.NOTEQUAL.toString() + boolean_out_scalar, boolean_scalar, boolean_out_scalar);
	}

	@Test
	public void testStringScalarWrite() {

		String string_scalar = "String Test.!";

		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", String.valueOf(string_scalar), output("a.scalar") };
		runTest(true, false, null, -1);

		String string_out_scalar = TestUtils.readDMLString(output(OUT_FILE));
		
		Assert.assertEquals("Values not equal: " + string_scalar + Opcodes.NOTEQUAL.toString() + string_out_scalar, string_scalar, string_out_scalar);
	}
	
	@Test
	public void testIntScalarRead() {

		int int_scalar = 464;
		
		setOutputBuffering(true);
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{"-args", String.valueOf(int_scalar), output("a.scalar")};
		runTest(true, false, null, -1);
		
		//int int_out_scalar = TestUtils.readDMLScalarFromHDFS(output(OUT_FILE)).get(new CellIndex(1,1)).intValue();
		//assertEquals("Values not equal: " + int_scalar + "!=" + int_out_scalar, int_scalar, int_out_scalar);
		
		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args", output("a.scalar"), "int" };
		
		ByteArrayOutputStream stdout =  runTest(true, false, null, -1);
		bufferContainsString(stdout, String.valueOf(int_scalar));
	}

	@Test
	public void testDoubleScalarRead() {

		double double_scalar = 464.5;
		
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", String.valueOf(double_scalar), output("a.scalar") };
		runTest(true, false, null, -1);
		
		//double double_out_scalar = TestUtils.readDMLScalarFromHDFS(output(OUT_FILE)).get(new CellIndex(1,1)).doubleValue();
		//assertEquals("Values not equal: " + double_scalar + "!=" + double_out_scalar, double_scalar, double_out_scalar);
		
		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args", output("a.scalar"), "double" };
		
		ByteArrayOutputStream stdout = runTest(true, false, null, -1);
		bufferContainsString(stdout, String.valueOf(double_scalar));
	}

	@Test
	public void testBooleanScalarRead() {

		boolean boolean_scalar = true;
		
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
			String.valueOf(boolean_scalar).toUpperCase(), output("a.scalar") };
		runTest(true, false, null, -1);

		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args", output("a.scalar"), "boolean" };
		
		// setExpectedStdOut(String.valueOf(boolean_scalar).toUpperCase());
		ByteArrayOutputStream stdout = runTest(true, false, null, -1);
		bufferContainsString(stdout, String.valueOf(boolean_scalar).toUpperCase());
	}

	@Test
	public void testStringScalarRead() {

		String string_scalar = "String Test.!";
		
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", String.valueOf(string_scalar), output("a.scalar") };
		runTest(true, false, null, -1);

		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args", output("a.scalar"), "string" };
		
		ByteArrayOutputStream stdout = runTest(true, false, null, -1);
		bufferContainsString(stdout, string_scalar);

	}
	
}
