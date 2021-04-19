package org.apache.sysds.test.functions.io.libsvm;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public abstract class ReadLIBSVMTestBase extends AutomatedTestBase {

  protected final static String TEST_DIR = "functions/io/libsvm/";
  protected final static String TEST_CLASS_DIR = TEST_DIR + ReadLIBSVMTest.class.getSimpleName() + "/";

  protected abstract String getTestClassDir();

  protected abstract String getTestName();

  @Override public void setUp() {
    addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"Rout"}));
  }
}
