package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class RoundTest extends AutomatedTestBase {

	private final static String TEST_NAME = "RoundTest";
	private final static String TEST_DIR = "functions/unary/scalar/";

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/unary/scalar/";
		availableTestConfigurations.put("RoundTest", new TestConfiguration("RoundTest", new String[] { "scalar" }));
	}
	
	@Test
	public void testRound() {
		TestConfiguration config = getTestConfiguration(TEST_NAME);

		double scalar = 10.7;
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", Double.toString(scalar), 
				                        HOME + OUTPUT_DIR + "scalar" };

		loadTestConfiguration("RoundTest");
		
		long seed = System.nanoTime();
		double roundScalar = Math.round(scalar);

		writeExpectedScalar("scalar", roundScalar);
		
		runTest(true, false, null, -1);
		
		HashMap<CellIndex, Double> map = readDMLScalarFromHDFS("scalar");
		double dmlvalue = map.get(new CellIndex(1,1));
		
		if ( dmlvalue != roundScalar ) {
			throw new RuntimeException("Values mismatch: DMLvalue " + dmlvalue + " != ExpectedValue " + roundScalar);
		}
		
		//compareResults();
	}
	
	
}
