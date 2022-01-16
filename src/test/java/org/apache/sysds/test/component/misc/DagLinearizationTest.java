package org.apache.sysds.test.component.misc;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

import static org.junit.Assert.fail;

public class DagLinearizationTest extends AutomatedTestBase  {

	private final String testName = "dag_linearization", testDir = "component/misc/";

	@Override
	public void setUp() {
		setOutputBuffering(true);
		disableConfigFile = true;
		TestUtils.clearAssertionInformation();
		addTestConfiguration(testName, new TestConfiguration(testDir, testName));
	}

	private String getPath(String filename) {
		return SCRIPT_DIR + "/" + testDir + filename;
	}

	@Test
	public void testMatrixMultSameOutput() {
		try {
			fullDMLScriptName = getPath("MatrixMult.dml");
			loadTestConfiguration(getTestConfiguration(testName));

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

}
