package com.ibm.bi.dml.test.integration.functions.data;



import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.BinaryMatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



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
public class WriteMMTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/data/";
		
		// positive tests
		availableTestConfigurations.put("SimpleTest", new TestConfiguration("WriteMMTest",
				new String[] { "a" }));
		availableTestConfigurations.put("ComplexTest", new TestConfiguration("WriteMMComplexTest",
				new String[] { "a" }));
		
		
		// negative tests
	}
	
	@Test
	public void testMM() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("SimpleTest");
		loadTestConfiguration("SimpleTest");
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrixMarket("a", a);
		
		runTest();
		
		compareResultsWithMM();
	}

	@Test
	public void testComplex() {
		
		int rows = 100;
		int cols = 100;
		
		TestConfiguration config = availableTestConfigurations.get("ComplexTest");
	
		
		loadTestConfiguration("ComplexTest");
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrixMarket("a", a);
		
		
		runTest();

			
		compareResultsWithMM();

	} 

} 
