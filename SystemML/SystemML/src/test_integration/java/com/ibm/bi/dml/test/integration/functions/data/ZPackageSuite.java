package com.ibm.bi.dml.test.integration.functions.data;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	FullReblockTest.class,
	FullStringInitializeTest.class,
	RandTest1.class,
	RandTest2.class,
	RandTest3.class,
	RandTest4.class,
	RandTest5.class,
	ReadMMTest.class,
	ReblockTest.class,
	SequenceTest.class,
	VariableTest.class,
	WriteMMTest.class,
	WriteTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
