package com.ibm.bi.dml.test.integration.functions.recompile;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;

/** Group together the tests in this package into a single suite so that the Maven build
 *  won't run two of them at once. */
@RunWith(Suite.class)
@Suite.SuiteClasses({
	BranchRemovalTest.class,
	CSVReadUnknownSizeTest.class,
	FunctionRecompileTest.class,
	IPAAssignConstantPropagationTest.class,
	IPAComplexAppendTest.class,
	IPAConstantPropagationTest.class,
	IPAPropagationSizeMultipleFunctionsTest.class,
	LiteralReplaceCastScalarReadTest.class,
	MultipleReadsIPATest.class,
	PredicateRecompileTest.class,
	RandJobRecompileTest.class,
	RandRecompileTest.class,
	RandSizeExpressionEvalTest.class,
	ReblockRecompileTest.class,
	RemoveEmptyPotpourriTest.class,
	RemoveEmptyRecompileTest.class,
	RewriteComplexMapMultChainTest.class,
	SparsityFunctionRecompileTest.class,
	SparsityRecompileTest.class
})


/** This class is just a holder for the above JUnit annotations. */
public class ZPackageSuite {

}
