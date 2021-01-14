package org.apache.sysds.test.functions.builtin;

import org.junit.Test;
import org.junit.Assert;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;

public class BuiltinBayesianOptimisationTest extends AutomatedTestBase {

	private final static String TEST_NAME = "bayesianOptimisation";
	//private final static String TEST_DIR = "./scripts/staging/functions/bayesian_optimisation/";
	private final static String TEST_DIR = "functions/builtin/";
    private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinBayesianOptimisationTest.class.getSimpleName() + "/";

	private final static int rows = 300;
	private final static int cols = 1;//20;

	@Override
	public void setUp()
	{
		//addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" })   );
        addTestConfiguration(TEST_DIR, TEST_NAME);
	}

    @Test
    public void testDodo(){

		ExecMode modeOld = setExecMode(ExecType.SPARK);

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
            String HOME = SCRIPT_DIR + TEST_DIR;
		//fullDMLScriptName = TEST_DIR + TEST_NAME;
			fullDMLScriptName = "./scripts/staging/bayesian_optimisation/bayesianOptimisation.dml"; //HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-args", "testFunc", input("X"), input("y"), "10" , output("R")};
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.8, -1);
			double[][] y = getRandomMatrix(rows, 1, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);


            String[] bounds = new String[] {"1", "1", "1"};
            String functionNameToOptimize = "name";

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		}
		finally {
			resetExecMode(modeOld);
		}
    }
}

