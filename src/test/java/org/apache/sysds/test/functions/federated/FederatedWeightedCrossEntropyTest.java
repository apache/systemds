
package org.apache.sysds.test.functions.federated;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedWeightedCrossEntropyTest extends AutomatedTestBase
{
  private final static Log LOG = LogFactory.getLog(FederatedWeightedCrossEntropyTest.class.getName());

  private final static String TEST_NAME = "FederatedWCeMMTest";
  private final static String TEST_DIR = "functions/federated/";
  private final static String TEST_CLASS_DIR = TEST_DIR + FederatedWeightedCrossEntropyTest.class.getSimpleName() + "/";

  private final static int blocksize = 1024;

  @Parameterized.Parameter()
  public int rows;
  @Parameterized.Parameter(1)
  public int cols;
  @Parameterized.Parameter(2)
  public int rank;
  @Parameterized.Parameter(3)
  public int epsilon_tolerance;

  // TODO: needed?
  // String testname, boolean sparse, boolean rewrites, boolean replication, ExecType instType
  // @Parameterized.Parameter(3)
  // public int replicate;
  // private final static double spSparse = ;
  // private final static double spDense = ;

  @Override
  public void setUp()
  {
    TestUtils.clearAssertionInformation();

    addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"Z"}));
  }

  @BeforeClass
  public static void init()
  {
    TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
  }

  @Test
  public void federatedWeightedCrossEntropySingleNode()
  {
    federatedWeightedCrossEntropy(ExecMode.SINGLE_NODE);
  }

  public void federatedWeightedCrossEntropy(ExecMode execMode)
  {
    // store the previous spark config and platform config to restore it after the test
    boolean spark_config_old = DMLScript.USE_LOCAL_SPARK_CONFIG;
    ExecMode platform_old = rtplatform;

    // TODO: ...

    getAndLoadTestConfiguration(TEST_NAME);
    String HOME = SCRIPT_DIR + TEST_DIR;

    int fed_rows = rows / 2;
    int fed_cols = cols;

    // generate dataset
    double[][] X1 = getRandomMatrix(fed_rows, fed_cols, 0, 1, 1, 3);
    double[][] X2 = getRandomMatrix(fed_rows, fed_cols, 0, 1, 1, 7);
    writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols));
    writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(fed_rows, fed_cols, blocksize, fed_rows * fed_cols));


    fullDMLScriptName = "";
    int port1 = getRandomAvailablePort();
    int port2 = getRandomAvailablePort();
    Thread thread1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
    Thread thread2 = startLocalFedWorkerThread(port2);

    rtplatform = execMode;
    if(rtplatform == ExecMode.SPARK)
    {
      DMLScript.USE_LOCAL_SPARK_CONFIG = true;
    }

    TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);

    // Run reference fml script with normal matrix
    fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
    // TODO: specify the program arguments according to the reference script
    programArgs = new String[] {};
    LOG.debug(runTest());

    // Run actual dml script with federated matrix
    fullDMLScriptName = HOME + TEST_NAME + ".dml";
    // TODO: specify the program arguments according to the test script
    programArgs = new String[] {};
    LOG.debug(runTest());

    // compare the results via files
    compareResults(epsilon_tolerance);

    // check that federated input files are still existing
    Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
    Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));

    TestUtils.shutdownThreads(thread1, thread2);

    DMLScript.USE_LOCAL_SPARK_CONFIG = spark_config_old;
    rtplatform = platform_old;

    assert false: "Not implemented yet!\n";
  }

}
