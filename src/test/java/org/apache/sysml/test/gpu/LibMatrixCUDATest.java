package org.apache.sysml.test.gpu;

import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.ScriptFactory;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;
import org.junit.*;

import java.util.Set;

/**
 * Unit tests for GPU methods
 */
public class LibMatrixCUDATest extends AutomatedTestBase {

	private final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
	private final static String TEST_NAME = "LibMatrixCUDATest";

	private final double THRESHOLD = 1e-9;
	private static SparkSession spark;

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@BeforeClass public static void beforeClass() {
		spark = createSystemMLSparkSession("LibMatrixCUDATest", "local");
	}

	// ****************************************************************
	// Unary Op Tests *************************************************
	// ****************************************************************

	final int[] unaryOpRowSizes = new int[]{ 1, 64, 130, 1024, 2049 };
	final int[] unaryOpColSizes = new int[]{ 1, 64, 130, 1024, 2049 };
	final double[] unaryOpSparsities = new double[] { 0.00, 0.3, 0.9 };
	final int unaryOpSeed = 42;


	@Test public void testSin() throws Exception {
		testUnaryOp("sin", "gpu_sin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testCos() throws Exception {
		testUnaryOp("cos", "gpu_cos", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testTan() throws Exception {
		testUnaryOp("tan", "gpu_tan", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAsin() throws Exception {
		testUnaryOp("asin", "gpu_asin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAcos() throws Exception {
		testUnaryOp("acos", "gpu_acos", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAtan() throws Exception {
		testUnaryOp("atan", "gpu_atan", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testExp() throws Exception {
		testUnaryOp("exp", "gpu_exp", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testLog() throws Exception {
		testUnaryOp("atan", "gpu_atan", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	// ****************************************************************
	// Unary Op Tests *************************************************
	// ****************************************************************



	/**
	 * Tests unary ops with a variety of matrix shapes and sparsities.
	 * Test is skipped for blocks of size 1x1.
	 * @param function name of the dml builtin unary op
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 * @param rows array of row sizes
	 * @param columns array of column sizes
	 * @param sparsities array of sparsities
	 * @param seed seed to use for random input matrix generation
	 */
	private void testUnaryOp(String function, String heavyHitterOpCode, int[] rows, int[] columns, double[] sparsities, int seed) {
		for (int i = 0; i < rows.length; i++) {
			for (int j = 0; j < columns.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					int row = rows[i];
					int column = columns[j];
					double sparsity = sparsities[k];
					// Skip the case of a scalar unary op
					if (row == 1 && column == 1)
						continue;

					System.out.println("Matrix of size [" + row + ", " + column + "], sparsity = " + sparsity);
					Matrix in1 = generateInputMatrix(spark, row, column, sparsity, seed);
					Matrix outCPU = runUnaryOpOnCPU(spark, function, in1);
					Matrix outGPU = runUnaryOpOnGPU(spark, function, in1);
					//assertHeavyHitterPresent(heavyHitterOpCode);
					assertEqualMatrices(outCPU, outGPU);
				}
			}
		}
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
	private Matrix generateInputMatrix(SparkSession spark, int m, int n, double sparsity, int seed) {
		// Generate a random matrix of size m * n
		MLContext genMLC = new MLContext(spark);
		String scriptStr;
		if (sparsity == 0.0) {
			scriptStr = "in1 = matrix(0, rows=" + m + ", cols=" + n + ")";
		} else {
			scriptStr = "in1 = rand(rows=" + m + ", cols=" + n + ", sparsity = " + sparsity + ", seed= " + seed +")";
		}
		Script generateScript = ScriptFactory.dmlFromString(scriptStr).out("in1");
		Matrix in1 = genMLC.execute(generateScript).getMatrix("in1");
		genMLC.close();
		return in1;
	}

	@AfterClass public static void afterClass() {
		spark.close();
	}

	@After public void tearDown() {
		super.tearDown();
	}

}