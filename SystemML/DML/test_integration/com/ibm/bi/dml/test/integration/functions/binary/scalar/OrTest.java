/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.scalar;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class OrTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/binary/scalar/";
		availableTestConfigurations.put("OrTest", new TestConfiguration("OrTest",
				new String[] { "left_1", "left_2", "left_3", "left_4", "right_1", "right_2", "right_3", "right_4" }));
	}
	
	@Test
	public void testOr() {
		loadTestConfiguration("OrTest");
		
		createHelperMatrix();
		writeExpectedHelperMatrix("left_1", 2);
		writeExpectedHelperMatrix("left_2", 2);
		writeExpectedHelperMatrix("left_3", 2);
		writeExpectedHelperMatrix("left_4", 1);
		writeExpectedHelperMatrix("right_1", 2);
		writeExpectedHelperMatrix("right_2", 2);
		writeExpectedHelperMatrix("right_3", 2);
		writeExpectedHelperMatrix("right_4", 1);
		
		runTest();
		
		compareResults();
	}

}
