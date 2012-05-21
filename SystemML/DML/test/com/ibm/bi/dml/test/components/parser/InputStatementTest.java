//package com.ibm.bi.dml.test.components.parser;
//
//import java.util.HashMap;
//
//import org.junit.Test;
//
//import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
//import com.ibm.bi.dml.test.integration.AutomatedTestBase;
//import com.ibm.bi.dml.test.integration.TestConfiguration;
//import com.ibm.bi.dml.test.utils.TestUtils;
//
//
//public class InputStatementTest extends AutomatedTestBase {
//
//	private final static String TEST_DIR = "functions/InputOutput/";
//	private final static String TEST_IO = "IOTestScipt";
//
//	@Override
//	public void setUp() {
//		//addTestConfiguration(TEST_IO, new TestConfiguration(TEST_DIR, TEST_IO, new String[] { "w", "h" }));
//	}
//	
//	@Test
//	public void testIO() {
//	
//		//TestConfiguration config = getTestConfiguration(TEST_GNMF);
//		
//		/* This is for running the junit test the new way, i.e., construct the arguments directly */
//		String IO_HOME = SCRIPT_DIR + TEST_DIR;
//		
//		/*
//		# $1 = "./test/scripts/functions/InputOutput/in/A.mtx"
//		# $2 = "./test/scripts/functions/InputOutput/in/B.mtx"
//		# $3 = 2
//		# $4 = 2
//		# $5 = "./test/scripts/functions/InputOutput/out/Aout-text.mtx"
//		# $6 = "./test/scripts/functions/InputOutput/out/Aout-binary.mtx" 
//		*/
//		dmlArgs = new String[]{"-f", IO_HOME + TEST_IO + "1.dml",
//				               "-args", IO_HOME + INPUT_DIR + "A.mtx", 
//				                        IO_HOME + INPUT_DIR + "B.mtx", 
//				                        "2",
//				                        "2",
//				                        IO_HOME + OUTPUT_DIR + "Aout-text.mtx", 
//				                        IO_HOME + OUTPUT_DIR + "Aout-binary.mtx"};
//		
//		dmlArgsDebug = new String[]{"-f", GNMF_HOME + TEST_IO + "1.dml", "-d",
//	                                "-args", GNMF_HOME + INPUT_DIR + "v", 
//	                                         GNMF_HOME + INPUT_DIR + "w", 
//	                                         GNMF_HOME + INPUT_DIR + "h", 
//	                                         Integer.toString(m), Integer.toString(n), Integer.toString(k), Integer.toString(maxiter),
//	                                         GNMF_HOME + OUTPUT_DIR + "w", 
//	                                         GNMF_HOME + OUTPUT_DIR + "h"};
//		
//		
//		boolean exceptionExpected = false;
//		int expectedNumberOfJobs = -1;
//		
//		/* GNMF must be run in the new way as GNMF.dml will be shipped */
//		runTest(true, exceptionExpected, null, -1); 
//		
//		//runRScript(true);
//		disableOutAndExpectedDeletion();
//
//		/*
//		HashMap<CellIndex, Double> hmWDML = readDMLMatrixFromHDFS("w");
//		HashMap<CellIndex, Double> hmHDML = readDMLMatrixFromHDFS("h");
//		HashMap<CellIndex, Double> hmWR = readRMatrixFromFS("w");
//		HashMap<CellIndex, Double> hmHR = readRMatrixFromFS("h");
//		HashMap<CellIndex, Double> hmWJava = TestUtils.convert2DDoubleArrayToHashMap(w);
//		HashMap<CellIndex, Double> hmHJava = TestUtils.convert2DDoubleArrayToHashMap(h);
//
//		TestUtils.compareMatrices(hmWDML, hmWR, 0.000001, "hmWDML", "hmWR");
//		TestUtils.compareMatrices(hmWDML, hmWJava, 0.000001, "hmWDML", "hmWJava");
//		TestUtils.compareMatrices(hmWR, hmWJava, 0.000001, "hmRDML", "hmWJava");
//		*/
//	}
//}
