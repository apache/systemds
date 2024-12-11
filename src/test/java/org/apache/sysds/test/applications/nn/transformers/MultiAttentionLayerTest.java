package org.apache.sysds.test.applications.nn.transformers;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class MultiAttentionLayerTest extends AutomatedTestBase {
    String TEST_NAME1 = "MultiAttentionForwardTest";
    private final static String TEST_DIR = "applications/nn/component/";
    
    @Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1));
	}
}
