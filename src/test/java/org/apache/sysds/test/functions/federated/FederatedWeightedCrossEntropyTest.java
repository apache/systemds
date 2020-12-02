
package org.apache.sysds.test.functions.federated;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedWeightedCrossEntropyTest extends AutomatedTestBase
{
  private final static String TEST_NAME = "FederatedWCeMMTest";
  private final static String TEST_DIR = "functions/federated/";
  private final static String TEST_CLASS_DIR = TEST_DIR + FederatedWeightedCrossEntropyTest.class.getSimpleName() + "/";

  @Parameterized.Parameter()
  public int rows;
  @Parameterized.Parameter(1)
  public int cols;
  @Parameterized.Parameter(2)
  public int rank;

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

  @Test
  public void federatedWeightedCrossEntropySingleNode()
  {
    federatedWeightedCrossEntropy(ExecMode.SINGLE_NODE);
  }

  public void federatedWeightedCrossEntropy(ExecMode execMode)
  {
    assert false: "Not implemented yet!\n";
  }

}
