package org.apache.sysds.test.component.misc;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public class DagLinearizationTest extends AutomatedTestBase  {

	private final String
		testNames[] = {"matrixmult_dag_linearization", "csplineCG_dag_linearization",
						"linear_regression_dag_linearization"},
		testDir = "component/misc/";

	@Override
	public void setUp() {
		setOutputBuffering(true);
		disableConfigFile = true;
		TestUtils.clearAssertionInformation();
		for (String testname : testNames) {
			addTestConfiguration(testname, new TestConfiguration(testDir, testname));
		}
	}

	private String getPath(String filename) {
		return SCRIPT_DIR + "/" + testDir + filename;
	}

	@Test
	public void testMatrixMultSameOutput() {
		try {
			fullDMLScriptName = getPath("MatrixMult.dml");
			loadTestConfiguration(getTestConfiguration(testNames[0]));

			// Default arguments
			programArgs = new String[] {"-config", "", "-args", output("totalResult")};

			programArgs[1] = getPath("SystemDS-config-default.xml");
			System.out.println(runTest(null).toString());
			HashMap<MatrixValue.CellIndex, Double> totalResultTopo = readDMLMatrixFromOutputDir("totalResult");

			programArgs[1] = getPath("SystemDS-config-minintermediate.xml");
			System.out.println(runTest(null).toString());
			HashMap<MatrixValue.CellIndex, Double> totalResultMin = readDMLMatrixFromOutputDir("totalResult");

			TestUtils.compareMatrices(totalResultTopo, totalResultMin, 0, "topological", "minintermediate");
		} catch (Exception ex) {
			ex.printStackTrace();
			fail("Exception in execution: " + ex.getMessage());
		}
	}

	@Test
	public void testCsplineCGSameOutput() {
		try {
			int rows = 10;
			int cols = 1;
			int numIter = rows;

			loadTestConfiguration(getTestConfiguration(testNames[1]));

			List<String> proArgs = new ArrayList<>();
			proArgs.add("-config");
			proArgs.add("");
			proArgs.add("-nvargs");
			proArgs.add("X=" + input("X"));
			proArgs.add("Y=" + input("Y"));
			proArgs.add("K=" + output("K"));
			proArgs.add("O=" + output("pred_y"));
			proArgs.add("maxi=" + numIter);
			proArgs.add("inp_x=" + 4.5);

			fullDMLScriptName = SCRIPT_DIR + "applications/cspline/CsplineCG.dml";

			double[][] X = new double[rows][cols];

			// X axis is given in the increasing order
			for(int rid = 0; rid < rows; rid++) {
				for(int cid = 0; cid < cols; cid++) {
					X[rid][cid] = rid + 1;
				}
			}

			double[][] Y = getRandomMatrix(rows, cols, 0, 5, 1.0, -1);

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			// Default
			proArgs.set(1, getPath("SystemDS-config-default.xml"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			String outputTopo = runTest(true, EXCEPTION_NOT_EXPECTED, null, -1).toString();
			System.out.println(outputTopo);
			HashMap<MatrixValue.CellIndex, Double> predYTopo = readDMLMatrixFromOutputDir("pred_y");

			// Min Intermediate
			proArgs.set(1, getPath("SystemDS-config-minintermediate.xml"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			String outputMin = runTest(true, EXCEPTION_NOT_EXPECTED, null, -1).toString();
			System.out.println(outputMin);
			HashMap<MatrixValue.CellIndex, Double> predYMin = readDMLMatrixFromOutputDir("pred_y");

			TestUtils.compareMatrices(predYTopo, predYMin, Math.pow(10, -5), "topological", "minintermediate");

			outputTopo = outputTopo.split("SystemDS Statistics:")[0];
			outputMin = outputMin.split("SystemDS Statistics:")[0];
			assertEquals("Outputs do not match!", outputTopo, outputMin);
		} catch (Exception ex) {
			ex.printStackTrace();
			fail("Exception in execution: " + ex.getMessage());
		}
	}

	@Test
	public void testLinearRegressionSameOutput() {
		try {
			int rows = 100;
			int cols = 50;

			loadTestConfiguration(getTestConfiguration(testNames[2]));

			List<String> proArgs = new ArrayList<>();
			proArgs.add("-config");
			proArgs.add("");
			proArgs.add("-args");
			proArgs.add(input("v"));
			proArgs.add(input("y"));
			proArgs.add(Double.toString(Math.pow(10, -8)));
			proArgs.add(output("w"));

			fullDMLScriptName = SCRIPT_DIR + "applications/linear_regression/LinearRegression.dml";

			double[][] v = getRandomMatrix(rows, cols, 0, 1, 0.01, -1);
			double[][] y = getRandomMatrix(rows, 1, 1, 10, 1, -1);
			writeInputMatrixWithMTD("v", v, true);
			writeInputMatrixWithMTD("y", y, true);

			int expectedNumberOfJobs = 16;

			proArgs.set(1, getPath("SystemDS-config-default.xml"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);
			HashMap<MatrixValue.CellIndex, Double> wTopo = readDMLMatrixFromOutputDir("w");

			proArgs.set(1, getPath("SystemDS-config-minintermediate.xml"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);
			HashMap<MatrixValue.CellIndex, Double> wMin = readDMLMatrixFromOutputDir("w");

			TestUtils.compareMatrices(wTopo, wMin, Math.pow(10, -10), "topological", "minintermediate");
		} catch (Exception ex) {
			ex.printStackTrace();
			fail("Exception in execution: " + ex.getMessage());
		}
	}

}
