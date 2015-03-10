package com.ibm.bi.dml.test.integration.functions.unary.scalar;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	AbsTest.class,
	ACosTest.class,
	ASinTest.class,
	ATanTest.class,
	BooleanTest.class,
	CosTest.class,
	ExponentTest.class,
	NegationTest.class,
	NotTest.class,
	PrintTest.class,
	RoundTest.class,
	SinTest.class,
	SqrtTest.class,
	StopTest2.class,
	StopTestCtrlStr.class,
	StopTest.class,
	TanTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
