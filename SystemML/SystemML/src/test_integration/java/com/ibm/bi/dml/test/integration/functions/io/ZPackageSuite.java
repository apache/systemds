package com.ibm.bi.dml.test.integration.functions.io;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	FullDynWriteTest.class,
	IOTest1.class,
	IOTest2.class,
	IOTest3.class,
	IOTest4.class,
	IOTest5.class,
	ScalarIOTest.class,
	SeqParReadTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
