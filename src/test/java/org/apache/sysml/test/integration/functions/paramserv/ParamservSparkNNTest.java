package org.apache.sysml.test.integration.functions.paramserv;

import static org.apache.sysml.api.mlcontext.ScriptFactory.dmlFromFile;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.junit.Test;

public class ParamservSparkNNTest extends AutomatedTestBase {

	private static final String TEST_NAME1 = "paramserv-test";
	private static final String TEST_NAME2 = "paramserv-spark-worker-failed";
	private static final String TEST_NAME3 = "paramserv-spark-agg-service-failed";

	private static final String TEST_DIR = "functions/paramserv/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ParamservSparkNNTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
	}

	@Test
	public void testParamservBSPBatchDisjointContiguous() {
		runDMLTest(10, 3, Statement.PSUpdateType.BSP, Statement.PSFrequency.BATCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS);
	}

	@Test
	public void testParamservASPBatchDisjointContiguous() {
		// FIXME arbitrary error will occur. Dimensions mismatch matrix-matrix binary operations: [0x0 vs 1x512]
		runDMLTest(10, 3, Statement.PSUpdateType.ASP, Statement.PSFrequency.BATCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS);
	}

	@Test
	public void testParamservBSPEpochDisjointContiguous() {
		// FIXME arbitrary error will occur. Caused by: (SparseBlockCSR.java:467) java.lang.ArrayIndexOutOfBoundsException: 118
		runDMLTest(10, 3, Statement.PSUpdateType.BSP, Statement.PSFrequency.EPOCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS);
	}

	@Test
	public void testParamservASPEpochDisjointContiguous() {
		// FIXME arbitrary error will occur. Caused by: (SparseBlockCSR.java:467) java.lang.ArrayIndexOutOfBoundsException: 113
		runDMLTest(10, 3, Statement.PSUpdateType.ASP, Statement.PSFrequency.EPOCH, 16, Statement.PSScheme.DISJOINT_CONTIGUOUS);
	}

	@Test
	public void testParamservWorkerFailed() {
		runDMLTest(TEST_NAME2, true, DMLException.class, "Invalid indexing by name in unnamed list: worker_err.");
	}

	@Test
	public void testParamservAggServiceFailed() {
		runDMLTest(TEST_NAME3, true, DMLException.class, "Invalid indexing by name in unnamed list: agg_service_err.");
	}

	private void runDMLTest(String testname, boolean exceptionExpected, Class<?> expectedException, String errMessage) {
		programArgs = new String[] { "-explain" };
		DMLScript.RUNTIME_PLATFORM oldRtplatform = AutomatedTestBase.rtplatform;
		boolean oldUseLocalSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
		AutomatedTestBase.rtplatform = DMLScript.RUNTIME_PLATFORM.SPARK;
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			runTest(true, exceptionExpected, expectedException, errMessage, -1);
		} finally {
			AutomatedTestBase.rtplatform = oldRtplatform;
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldUseLocalSparkConfig;
		}
	}

	private void runDMLTest(int epochs, int workers, Statement.PSUpdateType utype, Statement.PSFrequency freq, int batchsize, Statement.PSScheme scheme) {
		Script script = dmlFromFile(SCRIPT_DIR + TEST_DIR + TEST_NAME1 + ".dml").in("$mode", Statement.PSModeType.REMOTE_SPARK.toString())
			.in("$epochs", String.valueOf(epochs))
			.in("$workers", String.valueOf(workers))
			.in("$utype", utype.toString())
			.in("$freq", freq.toString())
			.in("$batchsize", String.valueOf(batchsize))
			.in("$scheme", scheme.toString());

		SparkConf conf = SparkExecutionContext.createSystemMLSparkConf().setAppName("ParamservSparkNNTest").setMaster("local[*]")
			.set("spark.driver.allowMultipleContexts", "true");
		JavaSparkContext sc = new JavaSparkContext(conf);
		MLContext ml = new MLContext(sc);
		ml.setStatistics(true);
		ml.execute(script);
		ml.resetConfig();
		sc.stop();
		ml.close();
	}
}
