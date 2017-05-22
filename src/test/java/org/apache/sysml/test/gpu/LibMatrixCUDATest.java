package org.apache.sysml.test.gpu;

import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.ScriptFactory;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;

import java.util.Set;

/**
 * Unit tests for GPU methods
 */
public class LibMatrixCUDATest extends AutomatedTestBase {

	private final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
	private final static String TEST_NAME = "LibMatrixCUDATest";

	private final double THRESHOLD = 1e-9;

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test public void testSin() throws Exception {
		SparkSession spark = createSystemMLSparkSession("LibMatrixCUDATest", "local");

		String function = "sin";
		String heavyHitterOpCode = "gpu_sin";

		int m = 100;
		int n = 100;
		double sparsity = 1.0;

		Matrix in1 = generateInputMatrix(spark, m, n, sparsity);
		Matrix outCPU = runUnaryOpOnCPU(spark, function, in1);
		assertHeavyHitterPresent(heavyHitterOpCode);
		Matrix outGPU = runUnaryOpOnGPU(spark, function, in1);
		assertEqualMatrices(outCPU, outGPU);

		spark.stop();

	}

	/**
	 * asserts that the expected op was executed
	 *
	 * @param heavyHitterOpCode opcode of the heavy hitter for the unary op
	 */
	private void assertHeavyHitterPresent(String heavyHitterOpCode) {
		Set<String> heavyHitterOpCodes = Statistics.getCPHeavyHitterOpCodes();
		Assert.assertTrue(heavyHitterOpCodes.contains(heavyHitterOpCode));
	}

	/**
	 * Asserts that the values in two matrices are in {@link LibMatrixCUDATest#THRESHOLD} of each other
	 *
	 * @param expected expected matrix
	 * @param actual   actual matrix
	 */
	private void assertEqualMatrices(Matrix expected, Matrix actual) {
		double[][] expected2D = expected.to2DDoubleArray();
		double[][] actual2D = actual.to2DDoubleArray();
		for (int i = 0; i < expected2D.length; i++) {
			Assert.assertArrayEquals(expected2D[i], actual2D[i], THRESHOLD);
		}
	}

	/**
	 * Runs the program for a unary op on the GPU
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param function dml builtin function string of the unary op
	 * @param in1      input matrix
	 * @return output matrix from running the unary op on the input matrix (on GPU)
	 */
	private Matrix runUnaryOpOnGPU(SparkSession spark, String function, Matrix in1) {
		String scriptStr2 = "out = " + function + "(in1)";
		MLContext gpuMLC = new MLContext(spark);
		gpuMLC.setGPU(true);
		gpuMLC.setForceGPU(true);
		gpuMLC.setStatistics(true);
		Script sinScript2 = ScriptFactory.dmlFromString(scriptStr2).in("in1", in1).out("out");
		Matrix outGPU = gpuMLC.execute(sinScript2).getMatrix("out");
		gpuMLC.close();
		return outGPU;
	}

	/**
	 * Runs the program for a unary op on the cpu
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param function dml builtin function string of the unary op
	 * @param in1      input matrix
	 * @return output matrix from running the unary op on the input matrix
	 */
	private Matrix runUnaryOpOnCPU(SparkSession spark, String function, Matrix in1) {
		String scriptStr1 = "out = " + function + "(in1)";
		MLContext cpuMLC = new MLContext(spark);
		Script sinScript = ScriptFactory.dmlFromString(scriptStr1).in("in1", in1).out("out");
		Matrix outCPU = cpuMLC.execute(sinScript).getMatrix("out");
		cpuMLC.close();
		return outCPU;
	}

	/**
	 * Generates a random input matrix with a given size and sparsity
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param m        number of rows
	 * @param n        number of columns
	 * @param sparsity sparsity (1 = completely dense, 0 = completely sparse)
	 * @return a random matrix with given size and sparsity
	 */
	private Matrix generateInputMatrix(SparkSession spark, int m, int n, double sparsity) {
		// Generate a random matrix of size m * n
		MLContext genMLC = new MLContext(spark);
		Script generateScript = ScriptFactory
				.dmlFromString("in1 = rand(rows=" + m + ", cols=" + n + ", sparsity = " + sparsity + ")").out("in1");
		Matrix in1 = genMLC.execute(generateScript).getMatrix("in1");
		genMLC.close();
		return in1;
	}

	@After public void tearDown() {
		super.tearDown();
	}

}