package org.apache.sysml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * Shared methods and fields from the different univariate stats test harness
 * classes.
 */
public abstract class UnivariateStatsBase extends AutomatedTestBase {

	/**
	 * Hard-coded path to the directory where this test keeps all of its files,
	 * generated and static
	 */
	protected final static String TEST_DIR = "applications/descriptivestats/";

	/** Fudge factor for comparisons against R's output */
	protected final static double epsilon = 0.000000001;

	/**
	 * Number of rows in the primary input to each aggregate ("V" in the test
	 * script, "X" in the SystemML documentation)
	 */
	protected final static int rows1 = 2000;

	/**
	 * Number of rows in the secondary input to aggregates that take two input
	 * vectors ("P" in the test script and the documentation)
	 */
	protected final static int rows2 = 5;

	/**
	 * Sizes of inputs used in tests, measured as number of rows in primary
	 * input vector. DIV4=divisible by 4; DIV4P1=divisible by 4 plus 1
	 */
	protected enum SIZE {
		DIV4(2000), DIV4P1(2001), DIV4P2(2002), DIV4P3(2003);
		int size = -1;

		SIZE(int s) {
			size = s;
		}
	};

	/** Ranges of values passed to aggregates in different tests. */
	protected enum RANGE {
		NEG(-255, -2), MIXED(-200, 200), POS(2, 255);
		double min, max;

		RANGE(double mn, double mx) {
			min = mn;
			max = mx;
		}
	};

	/**
	 * Actual sparsity values used in the "dense" and "sparse" variants of the
	 * tests.
	 */
	protected enum SPARSITY {
		SPARSE(0.3), DENSE(0.8);
		double sparsity;

		SPARSITY(double sp) {
			sparsity = sp;
		}
	};
	
	/** Shared setup code for test harness configurations */
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		
		addTestConfiguration("Scale", new TestConfiguration(TEST_DIR, "Scale",
				new String[] { "mean" + ".scalar", "std" + ".scalar",
						"se" + ".scalar", "var" + ".scalar", "cv" + ".scalar",
						/* "har", "geom", */
						"min" + ".scalar", "max" + ".scalar",
						"rng" + ".scalar", "g1" + ".scalar",
						"se_g1" + ".scalar", "g2" + ".scalar",
						"se_g2" + ".scalar", "out_minus", "out_plus",
						"median" + ".scalar", "quantile", "iqm" + ".scalar" }));
		addTestConfiguration("WeightedScaleTest", new TestConfiguration(
				TEST_DIR, "WeightedScaleTest", new String[] {
						"mean" + ".scalar", "std" + ".scalar",
						"se" + ".scalar", "var" + ".scalar", "cv" + ".scalar",
						/* "har", "geom", */
						"min" + ".scalar", "max" + ".scalar",
						"rng" + ".scalar", "g1" + ".scalar",
						"se_g1" + ".scalar", "g2" + ".scalar",
						"se_g2" + ".scalar", "out_minus", "out_plus",
						"median" + ".scalar", "quantile", "iqm" + ".scalar" }));
		addTestConfiguration("Categorical", new TestConfiguration(TEST_DIR,
				"Categorical", new String[] { "Nc", "R" + ".scalar", "Pc", "C",
						"Mode" })); // Indicate some file is scalar
		addTestConfiguration("WeightedCategoricalTest", new TestConfiguration(
				TEST_DIR, "WeightedCategoricalTest", new String[] { "Nc",
						"R" + ".scalar", "Pc", "C", "Mode" }));
	}

	/**
	 * Shared test driver for tests of univariate statistics over continuous,
	 * scaled, but not weighted, data
	 * 
	 * @param sz
	 *            size of primary input vector
	 * @param rng
	 *            range of randomly-generated values to use
	 * @param sp
	 *            sparsity of generated data
	 * @param rt
	 *            backend platform to test
	 */
	protected void testScaleWithR(SIZE sz, RANGE rng, SPARSITY sp,
			RUNTIME_PLATFORM rt) {

		RUNTIME_PLATFORM oldrt = rtplatform;
		rtplatform = rt;

		try {
			TestConfiguration config = getTestConfiguration("Scale");
			config.addVariable("rows1", sz.size);
			config.addVariable("rows2", rows2);

			// This is for running the junit test the new way, i.e., construct
			// the arguments directly
			String S_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = S_HOME + "Scale" + ".dml";
			programArgs = new String[] { "-args",
					S_HOME + INPUT_DIR + "vector", Integer.toString(sz.size),
					S_HOME + INPUT_DIR + "prob", Integer.toString(rows2),
					S_HOME + OUTPUT_DIR + "mean", S_HOME + OUTPUT_DIR + "std",
					S_HOME + OUTPUT_DIR + "se", S_HOME + OUTPUT_DIR + "var",
					S_HOME + OUTPUT_DIR + "cv", S_HOME + OUTPUT_DIR + "min",
					S_HOME + OUTPUT_DIR + "max", S_HOME + OUTPUT_DIR + "rng",
					S_HOME + OUTPUT_DIR + "g1", S_HOME + OUTPUT_DIR + "se_g1",
					S_HOME + OUTPUT_DIR + "g2", S_HOME + OUTPUT_DIR + "se_g2",
					S_HOME + OUTPUT_DIR + "median",
					S_HOME + OUTPUT_DIR + "iqm",
					S_HOME + OUTPUT_DIR + "out_minus",
					S_HOME + OUTPUT_DIR + "out_plus",
					S_HOME + OUTPUT_DIR + "quantile" };
			fullRScriptName = S_HOME + "Scale" + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + S_HOME + INPUT_DIR
					+ " " + S_HOME + EXPECTED_DIR;

			loadTestConfiguration(config);

			long seed1 = System.currentTimeMillis();
			long seed2 = System.currentTimeMillis();
			double[][] vector = getRandomMatrix(sz.size, 1, rng.min, rng.max,
					sp.sparsity, seed1);
			double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, seed2);
			System.out.println("seeds: " + seed1 + " " + seed2);

			writeInputMatrix("vector", vector, true);
			writeInputMatrix("prob", prob, true);

			// Expected number of jobs:
			// Reblock - 1 job
			// While loop iteration - 10 jobs
			// Final output write - 1 job
			//
			// boolean exceptionExpected = false;
			// int expectedNumberOfJobs = 12;
			// runTest(exceptionExpected, null, expectedNumberOfJobs);
			runTest(true, false, null, -1);

			runRScript(true);
			// disableOutAndExpectedDeletion();

			for (String file : config.getOutputFiles()) {
				// NOte that some files do not contain matrix, but just a single
				// scalar value inside
				HashMap<CellIndex, Double> dmlfile;
				HashMap<CellIndex, Double> rfile;
				if (file.endsWith(".scalar")) {
					file = file.replace(".scalar", "");
					dmlfile = readDMLScalarFromHDFS(file);
					rfile = readRScalarFromFS(file);
				} else {
					dmlfile = readDMLMatrixFromHDFS(file);
					rfile = readRMatrixFromFS(file);
				}
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file
						+ "-DML", file + "-R");
			}
		} finally {
			rtplatform = oldrt;
		}
	}

	/**
	 * Shared test driver for tests of univariate statistics over continuous,
	 * scaled, and weighted data
	 * 
	 * @param sz
	 *            size of primary input vector
	 * @param rng
	 *            range of randomly-generated values to use
	 * @param sp
	 *            sparsity of generated data
	 * @param rt
	 *            backend platform to test
	 */
	protected void testWeightedScaleWithR(SIZE sz, RANGE rng, SPARSITY sp,
			RUNTIME_PLATFORM rt) {

		RUNTIME_PLATFORM oldrt = rtplatform;
		rtplatform = rt;

		try {
			TestConfiguration config = getTestConfiguration("WeightedScaleTest");
			config.addVariable("rows1", sz.size);
			config.addVariable("rows2", rows2);

			// This is for running the junit test the new way, i.e., construct
			// the arguments directly
			String S_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = S_HOME + "WeightedScaleTest" + ".dml";
			programArgs = new String[] { "-args",
					S_HOME + INPUT_DIR + "vector", Integer.toString(sz.size),
					S_HOME + INPUT_DIR + "weight", S_HOME + INPUT_DIR + "prob",
					Integer.toString(rows2), S_HOME + OUTPUT_DIR + "mean",
					S_HOME + OUTPUT_DIR + "std", S_HOME + OUTPUT_DIR + "se",
					S_HOME + OUTPUT_DIR + "var", S_HOME + OUTPUT_DIR + "cv",
					S_HOME + OUTPUT_DIR + "min", S_HOME + OUTPUT_DIR + "max",
					S_HOME + OUTPUT_DIR + "rng", S_HOME + OUTPUT_DIR + "g1",
					S_HOME + OUTPUT_DIR + "se_g1", S_HOME + OUTPUT_DIR + "g2",
					S_HOME + OUTPUT_DIR + "se_g2",
					S_HOME + OUTPUT_DIR + "median",
					S_HOME + OUTPUT_DIR + "iqm",
					S_HOME + OUTPUT_DIR + "out_minus",
					S_HOME + OUTPUT_DIR + "out_plus",
					S_HOME + OUTPUT_DIR + "quantile" };
			fullRScriptName = S_HOME + "WeightedScaleTest" + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + S_HOME + INPUT_DIR
					+ " " + S_HOME + EXPECTED_DIR;

			loadTestConfiguration(config);

			createHelperMatrix();
			double[][] vector = getRandomMatrix(sz.size, 1, rng.min, rng.max,
					sp.sparsity, System.currentTimeMillis());
			double[][] weight = getRandomMatrix(sz.size, 1, 1, 10, 1,
					System.currentTimeMillis());
			OrderStatisticsTest.round(weight);
			double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1,
					System.currentTimeMillis());

			writeInputMatrix("vector", vector, true);
			writeInputMatrix("weight", weight, true);
			writeInputMatrix("prob", prob, true);

			//
			// Expected number of jobs:
			// Reblock - 1 job
			// While loop iteration - 10 jobs
			// Final output write - 1 job

			runTest(true, false, null, -1);

			runRScript(true);
			// disableOutAndExpectedDeletion();

			for (String file : config.getOutputFiles()) {
				// NOte that some files do not contain matrix, but just a single
				// scalar value inside
				HashMap<CellIndex, Double> dmlfile;
				HashMap<CellIndex, Double> rfile;
				if (file.endsWith(".scalar")) {
					file = file.replace(".scalar", "");
					dmlfile = readDMLScalarFromHDFS(file);
					rfile = readRScalarFromFS(file);
				} else {
					dmlfile = readDMLMatrixFromHDFS(file);
					rfile = readRMatrixFromFS(file);
				}
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file
						+ "-DML", file + "-R");
			}
		} finally {
			// reset runtime platform
			rtplatform = oldrt;
		}
	}

}
