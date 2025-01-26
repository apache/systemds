package org.apache.sysds.test.component.matrix;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.LibMatrixDatagen;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.RandomMatrixGenerator;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.*;

public class LibMatrixDatagenTest {
	protected static final Log LOG = LogFactory.getLog(LibMatrixDatagenTest.class.getName());

	@Test
	public void testGenerateUniformMatrixPhilox() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.CB_UNIFORM, 10, 10, 10, 1, 0., 1.);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 0L);
		for(int i = 0; i < 10; i++) {
			for(int j = 0; j < 10; j++) {
				assertTrue("Value: " + mb.get(i, j) + "needs to be less than 1", mb.get(i, j) < 1);
				assertTrue("Value: " + mb.get(i, j) + "needs to be greater than 0", mb.get(i, j) > 0);
			}
		}
	}

	@Test
	public void testGenerateNormalMatrixPhilox() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.CB_NORMAL, 1000, 1000, 1000 * 1000, 1);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 123123123123L);
		double mean = mb.mean();
		double[] bv = mb.getDenseBlockValues();
		double variance = Arrays.stream(bv).map(x -> Math.pow(x - mean, 2)).sum() / bv.length;
		assertEquals("Mean should be 0", 0, mean, 0.01);
		assertEquals("Variance should be 1", 1, variance, 0.001);
	}

	@Test
	public void testGenerateUniformMatrixPhiloxShouldHaveGoodStatistics() {
		MatrixBlock mb = new MatrixBlock();
		RandomMatrixGenerator rgen = new RandomMatrixGenerator(RandomMatrixGenerator.PDF.CB_UNIFORM, 1000, 1000, 100, 1, 0., 1.);
		LibMatrixDatagen.generateRandomMatrix(mb, rgen, null, 0L);

		double mean = mb.mean();
		assertEquals("Mean should be 0.5", 0.5, mean, 0.001);

		double[] bv = mb.getDenseBlockValues();
		assertEquals(1000 * 1000, bv.length);
		double variance = Arrays.stream(bv).map(x -> Math.pow(x - mean, 2)).sum() / bv.length;
		assertEquals("Variance should be 1", 0.0833, variance, 0.001);
	}
}
