package com.ibm.bi.dml.test.integration.applications;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
  ApplyTransformTest.class,
  ArimaTest.class,
  GNMFTest.class,
  PyDMLGNMFTest.class,
  PyDMLLinearRegressionTest.class,
  HITSTest.class,
  L2SVMTest.class,
  LinearLogRegTest.class,
  LinearRegressionTest.class,
  MultiClassSVMTest.class,
  //NMFCVTrainTest.class,
  PageRankTest.class,
  WelchTTest.class,
  
  GLMTest.class,
  ID3Test.class,
  MDABivariateStatsTest.class,
  NaiveBayesTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
