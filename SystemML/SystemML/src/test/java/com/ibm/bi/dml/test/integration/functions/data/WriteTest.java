/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.data;

import static org.junit.Assert.*;

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
public class WriteTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TEST_DIR = "functions/data/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		
		// positive tests
		addTestConfiguration("TextTest", new TestConfiguration(TEST_DIR, "WriteTest", new String[] { "a" }));
		addTestConfiguration("BinaryTest", new TestConfiguration(TEST_DIR, "WriteTest", new String[] { "a" }));
		addTestConfiguration("WriteTwiceTest", new TestConfiguration(TEST_DIR, "WriteTwiceTest", new String[] { "b", "c" }));
		
		// negative tests
	}
	
	@Test
	public void testText() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("TextTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "text");	
		loadTestConfiguration("TextTest");
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrix("a", a);
		
		runTest();
		
		compareResults();
	}
	
	@Test
	public void testBinary() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("BinaryTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		config.addVariable("format", "binary");
		loadTestConfiguration("BinaryTest");
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
				
		runTest();
		
		//compareResults();
		
		BinaryMatrixCharacteristics matrix = TestUtils.readBlocksFromSequenceFile(baseDirectory + OUTPUT_DIR + "a",1000,1000);
		//BinaryMatrixCharacteristics matrix = TestUtils.readBlocksFromSequenceFile(baseDirectory + OUTPUT_DIR + "a",1000,100);
		assertEquals(rows, matrix.getRows());
		assertEquals(cols, matrix.getCols());
		double[][] matrixValues = matrix.getValues();
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				assertEquals(i + "," + j, a[i][j], matrixValues[i][j], 0);
			}
		}
	}
	
	@Test
	public void testWriteTwice() {
		int rows = 10;
		int cols = 10;
		
		TestConfiguration config = availableTestConfigurations.get("WriteTwiceTest");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration("WriteTwiceTest");
		
		double[][] a = getRandomMatrix(rows, cols, -1, 1, 0.7, System.currentTimeMillis());
		writeInputMatrixWithMTD("a", a, false, new MatrixCharacteristics(rows,cols,1000,1000));
		writeExpectedMatrix("b", a);
		writeExpectedMatrix("c", a);
		
		
		runTest();

			
		compareResults();

	}

}
