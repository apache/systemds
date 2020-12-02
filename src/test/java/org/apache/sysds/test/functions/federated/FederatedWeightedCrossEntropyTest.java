
package org.apache.sysds.test.functions.federated;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FederatedWeightedCrossEntropyTest extends AutomatedTestBase
{
  private final static String TEST_NAME = "FederatedWCeMMTest";
  private final static String TEST_DIR = "functions/federated/";
  private final static String TEST_CLASS_DIR = FederatedWeightedCrossEntropyTest.class.getSimpleName() + "/";

  // TODO: define these things according to the test data
  // private final static int rows = ;
  // private final static int cols = ;
  // private final static int rank = ;
  // private final static double spSparse = ;
  // private final static double spDense = ;

  @Override
  public void setUp()
  {
    TestUtils.clearAssertionInformation();


    // TODO: check if the following lines are correct
    //        replace the TODO of the next line of code to the correct letter
    addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"TODO"}));

    if(TEST_CACHE_ENABLED)
    {
      setOutAndExpectedDeletionDisabled(true);
    }
    assert false: "Not implemented yet!\n";
  }
}
