/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class RoundTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "RoundTest";
	private final static String TEST_DIR = "functions/unary/scalar/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("RoundTest", new TestConfiguration(TEST_DIR,"RoundTest", new String[] { "scalar" }));
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

		loadTestConfiguration(config);
		
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
