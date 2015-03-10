package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	ConditionalValidateTest.class,
	DataTypeCastingTest.class,
	DataTypeChangeTest.class,
	FunctionInliningTest.class,
	IPALiteralReplacementTest.class,
	IPAScalarRecursionTest.class,
	IPAUnknownRecursionTest.class,
	LongOverflowTest.class,
	NrowNcolStringTest.class,
	NrowNcolUnknownCSVReadTest.class,
	ReadAfterWriteTest.class,
	ScalarAssignmentTest.class,
	ValueTypeAutoCastingTest.class,
	ValueTypeCastingTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
