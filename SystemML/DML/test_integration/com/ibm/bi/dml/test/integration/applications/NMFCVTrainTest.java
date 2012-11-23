package com.ibm.bi.dml.test.integration.applications;

//import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



public class NMFCVTrainTest extends AutomatedTestBase {

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
