package org.tugraz.sysds.test.functions.builtin;

import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class BuiltinKmeansTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "kmeans";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinKmeansTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private final static int rows = 3972;
	private final static int cols = 972;
	private final static double spSparse = 0.3;
	private final static double spDense = 0.7;
	private final static double max_iter = 10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testKMeansDenseBinSingleRewritesCP() {
		runKMeansTest(false, 2, 1, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseBinSingleRewritesCP() {
		runKMeansTest(true,2, 1, true,  LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseBinSingleCP() {
		runKMeansTest(false,2, 1, false,  LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseBinSingleCP() {
		runKMeansTest(true, 2, 1, false, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseBinMultiRewritesCP() {
		runKMeansTest(false, 2, 10, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseBinMultiRewritesCP() {
		runKMeansTest(true, 2, 10, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseBinMultiCP() {
		runKMeansTest(false, 2, 10, false, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseBinMultiCP() {
		runKMeansTest(true, 2, 10, false, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseMulSingleRewritesCP() {
		runKMeansTest(false, 20, 1, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseMulSingleRewritesCP() {
		runKMeansTest(true, 20, 1, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseMulSingleCP() {
		runKMeansTest(false, 20, 1, false, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseMulSingleCP() {
		runKMeansTest(true, 20, 1, false, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseMulMultiRewritesCP() {
		runKMeansTest( false, 20, 10, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseMulMultiRewritesCP() {
		runKMeansTest(true, 20, 10, true, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansDenseMulMultiCP() {
		runKMeansTest(false, 20, 10, false, LopProperties.ExecType.CP);
	}

	@Test
	public void testKMeansSparseMulMultiCP() {
		runKMeansTest(true, 20, 10, false, LopProperties.ExecType.CP);
	}
	
	private void runKMeansTest(boolean sparse, int centroids, int runs,
		boolean rewrites, LopProperties.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);

		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-explain", "-stats",
					"-nvargs", "X=" + input("X"), "Y=" + output("Y"), "C=" + output("C"),
					"k=" + centroids, "runs=" + runs,
					"eps=" + eps, "max_iter=" + max_iter};

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 714);
			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}
}
