package com.ibm.bi.dml.test.integration.functions.aggregate;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
  ColSumTest.class,
  LengthTest.class,
  MaxTest.class,
  MinTest.class,
  NColTest.class,
  NRowTest.class,
  ProdTest.class,
  RowSumTest.class,
  SumTest.class,
  TraceTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
