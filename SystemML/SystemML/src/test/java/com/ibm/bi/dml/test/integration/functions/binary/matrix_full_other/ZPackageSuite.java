package com.ibm.bi.dml.test.integration.functions.binary.matrix_full_other;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	
	FullMatrixMultiplicationUltraSparseTest.class,
	FullIntegerDivisionTest.class,
	FullMatrixMultiplicationTest.class,
	FullMatrixMultiplicationTransposeSelfTest.class,
	FullMinMaxComparisonTest.class,
	FullPowerTest.class,
	FullPPredMatrixTest.class,
	FullPPredScalarLeftTest.class,
	FullPPredScalarRightTest.class

})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
