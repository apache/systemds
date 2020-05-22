package org.apache.sysds.test.functions.privacy;

import java.util.Arrays;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.apache.sysds.common.Types;
import static java.lang.Thread.sleep;

public class FederatedWorkerHandlerTest extends AutomatedTestBase {

    private static final String TEST_DIR = "functions/federated/";
    private static final String TEST_DIR_SCALAR = TEST_DIR + "matrix_scalar/";
    private final static String TEST_CLASS_DIR = TEST_DIR + FederatedWorkerHandlerTest.class.getSimpleName() + "/";
    private final static String TEST_CLASS_DIR_SCALAR = TEST_DIR_SCALAR + FederatedWorkerHandlerTest.class.getSimpleName() + "/";
    private static final String TEST_PROG_SCALAR_ADDITION_MATRIX = "ScalarAdditionFederatedMatrix";
    private final static String AGGREGATION_TEST_NAME = "FederatedSumTest";
    private final static String TRANSFER_TEST_NAME = "FederatedRCBindTest";
    private static final String FEDERATED_WORKER_HOST = "localhost";
	private static final int FEDERATED_WORKER_PORT = 1222;
    
    private final static int blocksize = 1024;
    private int rows = 10;
    private int cols = 10;
    

    @Override
    public void setUp() {
        TestUtils.clearAssertionInformation();
        addTestConfiguration("scalar", new TestConfiguration(TEST_CLASS_DIR_SCALAR, TEST_PROG_SCALAR_ADDITION_MATRIX, new String [] {"R"}));
        addTestConfiguration("aggregation", new TestConfiguration(TEST_CLASS_DIR, AGGREGATION_TEST_NAME, new String[] {"S.scalar", "R", "C"}));
        addTestConfiguration("transfer", new TestConfiguration(TEST_CLASS_DIR, TRANSFER_TEST_NAME, new String[] {"R", "C"}));
    }

    @Test
    public void scalarPrivateTest(){
        scalarTest(PrivacyLevel.Private, DMLException.class);
    }

    @Test
    public void scalarPrivateAggregationTest(){
        scalarTest(PrivacyLevel.PrivateAggregation, DMLException.class);
    }

    @Test
    public void scalarNonePrivateTest(){
        scalarTest(PrivacyLevel.None, null);
    }

    private void scalarTest(PrivacyLevel privacyLevel, Class<?> expectedException){
        getAndLoadTestConfiguration("scalar");

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
        
        PrivacyConstraint pc = new PrivacyConstraint(privacyLevel);
        writeInputMatrixWithMTD("M", m, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols), pc);

		int s = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] + s;
			}
        }
        if (expectedException == null)
		    writeExpectedMatrix("R", r);

		runGenericScalarTest(TEST_PROG_SCALAR_ADDITION_MATRIX, s, expectedException);
    }


    private void runGenericScalarTest(String dmlFile, int s, Class<?> expectedException)
	{
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;
		try {
			// we need the reference file to not be written to hdfs, so we get the correct format
			rtplatform = Types.ExecMode.SINGLE_NODE;
			if (rtplatform == Types.ExecMode.SPARK) {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			}
			programArgs = new String[] {"-w", Integer.toString(FEDERATED_WORKER_PORT)};
			t = new Thread(() -> runTest(true, false, null, -1));
			t.start();
			sleep(FED_WORKER_WAIT);
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR_SCALAR + dmlFile + ".dml";
			programArgs = new String[]{"-args",
					TestUtils.federatedAddress(FEDERATED_WORKER_HOST, FEDERATED_WORKER_PORT, input("M")),
					Integer.toString(rows), Integer.toString(cols),
					Integer.toString(s),
                    output("R")};
            boolean exceptionExpected = (expectedException != null);
			runTest(true, exceptionExpected, expectedException, -1);

            if ( !exceptionExpected )
			    compareResults();
		} catch (InterruptedException e) {
			e.printStackTrace();
			assert (false);
		} finally {
			rtplatform = platformOld;
			TestUtils.shutdownThread(t);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
    }

    @Test
	public void aggregatePrivateTest() {
		federatedSum(Types.ExecMode.SINGLE_NODE, PrivacyLevel.Private, DMLException.class);
    }
    
    @Test
	public void aggregatePrivateAggregationTest() {
		federatedSum(Types.ExecMode.SINGLE_NODE, PrivacyLevel.PrivateAggregation, null);
    }
    
    @Test
	public void aggregateNonePrivateTest() {
		federatedSum(Types.ExecMode.SINGLE_NODE, PrivacyLevel.None, null);
	}
    
    public void federatedSum(Types.ExecMode execMode, PrivacyLevel privacyLevel, Class<?> expectedException) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;

		getAndLoadTestConfiguration("aggregation");
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] A = getRandomMatrix(rows, cols, -10, 10, 1, 1);
		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols), new PrivacyConstraint(privacyLevel));
		int port = getRandomAvailablePort();
		t = startLocalFedWorker(port);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix for Row/Col sum
		fullDMLScriptName = HOME + AGGREGATION_TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-args", input("A"), input("A"), expected("R"), expected("C")};
		runTest(true, false, null, -1);

		// write expected sum
		double sum = 0;
		for(double[] doubles : A) {
			sum += Arrays.stream(doubles).sum();
		}
        sum *= 2;
        
        if ( expectedException == null )
		    writeExpectedScalar("S", sum);

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get("aggregation");
		loadTestConfiguration(config);
		fullDMLScriptName = HOME + AGGREGATION_TEST_NAME + ".dml";
		programArgs = new String[] {"-args", "\"localhost:" + port + "/" + input("A") + "\"", Integer.toString(rows),
			Integer.toString(cols), Integer.toString(rows * 2), output("S"), output("R"), output("C")};

		runTest(true, (expectedException != null), expectedException, -1);

        // compare all sums via files
        if ( expectedException == null )
		    compareResults(1e-11);

		TestUtils.shutdownThread(t);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
    }

    @Test
	public void transferPrivateTest() {
		federatedRCBind(Types.ExecMode.SINGLE_NODE, PrivacyLevel.Private, DMLException.class);
    }
    
    @Test
	public void transferPrivateAggregationTest() {
		federatedRCBind(Types.ExecMode.SINGLE_NODE, PrivacyLevel.PrivateAggregation, DMLException.class);
    }
    
    @Test
	public void transferNonePrivateTest() {
		federatedRCBind(Types.ExecMode.SINGLE_NODE, PrivacyLevel.None, null);
	}
    
    public void federatedRCBind(Types.ExecMode execMode, PrivacyLevel privacyLevel, Class<?> expectedException) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;

		getAndLoadTestConfiguration("transfer");
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] A = getRandomMatrix(rows, cols, -10, 10, 1, 1);
		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols), new PrivacyConstraint(privacyLevel));

		int port = getRandomAvailablePort();
		t = startLocalFedWorker(port);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix for Row/Col sum
		fullDMLScriptName = HOME + TRANSFER_TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-args", input("A"), expected("R"), expected("C")};
		runTest(true, false, null, -1);

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get("transfer");
		loadTestConfiguration(config);
		fullDMLScriptName = HOME + TRANSFER_TEST_NAME + ".dml";
		programArgs = new String[] {"-args", "\"localhost:" + port + "/" + input("A") + "\"", Integer.toString(rows),
			Integer.toString(cols), output("R"), output("C")};

        runTest(true, (expectedException != null), expectedException, -1);

		// compare all sums via files
		if ( expectedException == null )
		    compareResults(1e-11);

		TestUtils.shutdownThread(t);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

}