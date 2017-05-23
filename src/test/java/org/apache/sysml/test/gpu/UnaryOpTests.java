package org.apache.sysml.test.gpu;

import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * Unit tests for GPU methods
 */
public class UnaryOpTests extends UnaryOpTestsBase {

	private final static String TEST_NAME = "UnaryOpTests";

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}


	// ****************************************************************
	// Unary Op Tests *************************************************
	// ****************************************************************

	@Test public void testSin() throws Exception {
		testUnaryOpMatrixOutput("sin", "gpu_sin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testCos() throws Exception {
		testUnaryOpMatrixOutput("cos", "gpu_cos", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testTan() throws Exception {
		testUnaryOpMatrixOutput("tan", "gpu_tan", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAsin() throws Exception {
		testUnaryOpMatrixOutput("asin", "gpu_asin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAcos() throws Exception {
		testUnaryOpMatrixOutput("acos", "gpu_acos", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAtan() throws Exception {
		testUnaryOpMatrixOutput("atan", "gpu_atan", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testExp() throws Exception {
		testUnaryOpMatrixOutput("exp", "gpu_exp", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testLog() throws Exception {
		testUnaryOpMatrixOutput("log", "gpu_log", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testSqrt() throws Exception {
		testUnaryOpMatrixOutput("sqrt", "gpu_sqrt", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testAbs() throws Exception {
		testUnaryOpMatrixOutput("abs", "gpu_abs", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testRound() throws Exception {
		testUnaryOpMatrixOutput("round", "gpu_round", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testFloor() throws Exception {
		testUnaryOpMatrixOutput("sqrt", "gpu_floor", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void testCeil() throws Exception {
		testUnaryOpMatrixOutput("ceil", "gpu_ceil", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}


	// ****************************************************************
	// Unary Op Tests *************************************************
	// ****************************************************************

}