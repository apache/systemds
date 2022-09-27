package org.apache.sysds.test.functions.io.compressed;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class ReadCompressedTest extends CompressedTestBase {

	protected abstract int getId();

	protected String getInputFileName() {
		return "comp_" + getId();
	}

	@Test
	public void testCSV_Sequential_CP1() {
		runTest(getId(), ExecMode.SINGLE_NODE, false);
	}

	@Test
	public void testCSV_Parallel_CP1() {
		runTest(getId(), ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void testCSV_Sequential_CP() {
		runTest(getId(), ExecMode.HYBRID, false);
	}

	@Test
	public void testCSV_Parallel_CP() {
		runTest(getId(), ExecMode.HYBRID, true);
	}

	@Test
	public void testCSV_SP() {
		runTest(getId(), ExecMode.SPARK, false);
	}

	protected String runTest(int testNumber, ExecMode platform, boolean parallel) {
		ExecMode oldPlatform = rtplatform;
		rtplatform = platform;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		boolean oldpar = CompilerConfig.FLAG_PARREADWRITE_TEXT;

		String output;
		try {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;

			TestConfiguration config = getTestConfiguration(getTestName());
			loadTestConfiguration(config);
			setOutputBuffering(true); // otherwise NPEs

			String HOME = SCRIPT_DIR + TEST_DIR;
			String inputMatrixNameNoExtension = HOME + INPUT_DIR + getInputFileName();
			String inputMatrixNameWithExtension = inputMatrixNameNoExtension + ".csv";
			String dmlOutput = output("dml.scalar");
			String rOutput = output("R.scalar");

			String sep = getId() == 2 ? ";" : ",";

			fullDMLScriptName = HOME + getTestName() + "_" + testNumber + ".dml";
			programArgs = new String[] {"-args", inputMatrixNameWithExtension, dmlOutput, sep};

			fullRScriptName = HOME + "csv_verify2.R";
			rCmd = "Rscript " + fullRScriptName + " " + inputMatrixNameNoExtension + ".single.csv " + rOutput;

			output = runTest(true, false, null, -1).toString();
			runRScript(true);

			double dmlScalar = TestUtils.readDMLScalar(dmlOutput);
			double rScalar = TestUtils.readRScalar(rOutput);
			TestUtils.compareScalars(dmlScalar, rScalar, eps);
		}
		finally {
			rtplatform = oldPlatform;
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldpar;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		return output;
	}
}
