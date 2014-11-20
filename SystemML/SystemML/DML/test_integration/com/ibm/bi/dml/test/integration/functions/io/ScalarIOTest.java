/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.io;

import static junit.framework.Assert.assertEquals;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ScalarIOTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "scalarIOTest";
	private final static String TEST_DIR = "functions/io/";
	private final static String OUT_FILE = SCRIPT_DIR + TEST_DIR + OUTPUT_DIR + "a.scalar";
	
	@Override
	public void setUp() {
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "a.scalar" })   );  
		//baseDirectory = SCRIPT_DIR + "functions/io/";
	}

	@Test
	public void testIntScalarWrite() {

		int int_scalar = 464;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(int_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);
		int int_out_scalar = TestUtils.readDMLScalarFromHDFS(OUT_FILE).get(new CellIndex(1,1)).intValue();
		assertEquals("Values not equal: " + int_scalar + "!=" + int_out_scalar, int_scalar, int_out_scalar);
		
		// Invoke the DML script that does computations and then writes scalar to HDFS
		fullDMLScriptName = HOME + "ScalarComputeWrite.dml";
		runTest(true, false, null, -1);
		int_out_scalar = TestUtils.readDMLScalarFromHDFS(OUT_FILE).get(new CellIndex(1,1)).intValue();
		assertEquals("Computation test for Integers failed: Values not equal: " + int_scalar + "!=" + int_out_scalar, int_scalar, int_out_scalar);
		
	}

	@Test
	public void testDoubleScalarWrite() {

		double double_scalar = 464.55;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(double_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);
		double double_out_scalar = TestUtils.readDMLScalarFromHDFS(OUT_FILE).get(new CellIndex(1,1)).doubleValue();
		assertEquals("Values not equal: " + double_scalar + "!=" + double_out_scalar, double_scalar, double_out_scalar);

		// Invoke the DML script that does computations and then writes scalar to HDFS
		fullDMLScriptName = HOME + "ScalarComputeWrite.dml";
		runTest(true, false, null, -1);
		double_out_scalar = TestUtils.readDMLScalarFromHDFS(OUT_FILE).get(new CellIndex(1,1)).doubleValue();
		assertEquals("Computation test for Integers failed: Values not equal: " + double_scalar + "!=" + double_out_scalar, double_scalar, double_out_scalar);
	}

	@Test
	public void testBooleanScalarWrite() {

		boolean boolean_scalar = true;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(boolean_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);

		boolean boolean_out_scalar = TestUtils.readDMLBoolean(OUT_FILE);
		
		assertEquals("Values not equal: " + boolean_scalar + "!=" + boolean_out_scalar, boolean_scalar, boolean_out_scalar);
	}

	@Test
	public void testStringScalarWrite() {

		String string_scalar = "String Test.!";
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(string_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);

		String string_out_scalar = TestUtils.readDMLString(OUT_FILE);
		
		assertEquals("Values not equal: " + string_scalar + "!=" + string_out_scalar, string_scalar, string_out_scalar);
	}
	
	@Test
	public void testIntScalarRead() {

		int int_scalar = 464;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(int_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);
		//int int_out_scalar = TestUtils.readDMLScalarFromHDFS(OUT_FILE).get(new CellIndex(1,1)).intValue();
		//assertEquals("Values not equal: " + int_scalar + "!=" + int_out_scalar, int_scalar, int_out_scalar);
		
		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args",
									 HOME + OUTPUT_DIR + "a.scalar",
									 "int"
									};
		
		setExpectedStdOut(String.valueOf(int_scalar));
		runTest(true, false, null, -1);
		
	}

	@Test
	public void testDoubleScalarRead() {

		double double_scalar = 464.5;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(double_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);
		//double double_out_scalar = TestUtils.readDMLScalarFromHDFS(OUT_FILE).get(new CellIndex(1,1)).doubleValue();
		//assertEquals("Values not equal: " + double_scalar + "!=" + double_out_scalar, double_scalar, double_out_scalar);
		
		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args",
									 HOME + OUTPUT_DIR + "a.scalar",
									 "double"
									};
		
		setExpectedStdOut(String.valueOf(double_scalar));
		runTest(true, false, null, -1);
		
	}

	@Test
	public void testBooleanScalarRead() {

		boolean boolean_scalar = true;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// TODO Niketan: Separate these as individual tests
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(boolean_scalar).toUpperCase(),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);

		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args",
									 HOME + OUTPUT_DIR + "a.scalar",
									 "boolean"
									};
		
		setExpectedStdOut(String.valueOf(boolean_scalar).toUpperCase());
		runTest(true, false, null, -1);
	}

	@Test
	public void testStringScalarRead() {

		String string_scalar = "String Test.!";
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + "ScalarWrite.dml";
		programArgs = new String[]{	"-args", 
									String.valueOf(string_scalar),
									HOME + OUTPUT_DIR + "a.scalar"
                				  };
		runTest(true, false, null, -1);

		// Invoke the DML script that reads the scalar and prints to stdout
		fullDMLScriptName = HOME + "ScalarRead.dml";
		programArgs = new String[] { "-args",
									 HOME + OUTPUT_DIR + "a.scalar",
									 "string"
									};
		
		setExpectedStdOut(String.valueOf(string_scalar));
		runTest(true, false, null, -1);
	}
	

}
