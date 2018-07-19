package org.apache.sysml.test.integration.functions.paramserv;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Test;

public class ParamservSparkNNTest extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-spark-nn-bsp-batch-dc";
	private static final String TEST_NAME2 = "paramserv-spark-nn-asp-batch-dc";
	private static final String TEST_NAME3 = "paramserv-spark-nn-bsp-epoch-dc";
	private static final String TEST_NAME4 = "paramserv-spark-nn-asp-epoch-dc";
	private static final String TEST_NAME5 = "paramserv-spark-worker-failed";
	private static final String TEST_NAME6 = "paramserv-spark-agg-service-failed";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservSparkNNTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {}));
	}

	@Test
	public void testParamservBSPBatchDisjointContiguous() {
		runDMLTest(TEST_NAME1, false, null, null);
	}

	@Test
	public void testParamservASPBatchDisjointContiguous() {
		// FIXME Dimensions mismatch matrix-matrix binary operations: [0x0 vs 1x512]
		runDMLTest(TEST_NAME2, false, null, null);
	}

	@Test
	public void testParamservBSPEpochDisjointContiguous() {
		// FIXME Caused by: (SparseBlockCSR.java:467) java.lang.ArrayIndexOutOfBoundsException: 118
		runDMLTest(TEST_NAME3, false, null, null);
	}

	@Test
	public void testParamservASPEpochDisjointContiguous() {
		// FIXME Caused by: (SparseBlockCSR.java:467) java.lang.ArrayIndexOutOfBoundsException: 113
		runDMLTest(TEST_NAME4, false, null, null);
	}

	@Test
	public void testParamservWorkerFailed() {
		runDMLTest(TEST_NAME5, true, DMLException.class, "Invalid indexing by name in unnamed list: worker_err.");
	}

	@Test
	public void testParamservAggServiceFailed() {
		runDMLTest(TEST_NAME6, true, DMLException.class, "Invalid indexing by name in unnamed list: agg_service_err.");
	}

	private void runDMLTest(String testname, boolean exceptionExpected, Class<?> expectedException, String errMessage) {
		DMLScript.RUNTIME_PLATFORM oldRtplatform = AutomatedTestBase.rtplatform;
		boolean oldUseLocalSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		AutomatedTestBase.rtplatform = DMLScript.RUNTIME_PLATFORM.SPARK;
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			programArgs = new String[] { "-explain" };
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			// The test is not already finished, so it is normal to have the NPE
			runTest(true, exceptionExpected, expectedException, errMessage, -1);
		} finally {
			AutomatedTestBase.rtplatform = oldRtplatform;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseLocalSparkConfig;
		}

	}

}
