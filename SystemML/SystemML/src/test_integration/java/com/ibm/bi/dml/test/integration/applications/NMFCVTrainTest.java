/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications;

//import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class NMFCVTrainTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "applications/nmf_cv_train/";
		availableTestConfigurations.put("NMFCVTrainTest", new TestConfiguration("NMFCVTrainTest",
				new String[] { "nmf.wb.result", "nmf.hc.result" }));
	}
	
	/*
	@Test
	public void testNMFCVTrain() {
		loadTestConfiguration("NMFCVTrainTest");
		
		createRandomMatrix("x", 5, 5, 0, 1, 0.8, -1);
		createRandomMatrix("y", 5, 5, 0, 1, 0.8, -1);
		createRandomMatrix("z", 5, 5, 0, 1, 0.8, -1);
		
		runTest();
		
		checkForResultExistence();
	}
	*/
}
