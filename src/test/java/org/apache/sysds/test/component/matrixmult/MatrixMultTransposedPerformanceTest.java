package org.apache.sysds.test.component.matrixmult;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Assert;
import org.junit.Test;

public class MatrixMultTransposedPerformanceTest {
	private final int m = 100;
	private final int n = 100;
	private final int k = 100;


	@Test
	public void testPerf_1_NoTransA_TransB() {
		System.out.println("Case: C = A %*% t(B)");
		runTest(false, true);
		System.out.println();
	}

	@Test
	public void testPerf_2_TransA_NoTransB() {
		System.out.println("Case: C = t(A) %*% B");
		runTest(true, false);
		System.out.println();
	}

	@Test
	public void testPerf_3_TransA_TransB() {
		System.out.println("Case: C = t(A) %*% t(B)");
		runTest(true, true);
	}

	private void runTest(boolean tA, boolean tB) {
		int REP = 5000;

		// setup Dimensions
		int rowsA = tA ? k : m;
		int colsA = tA ? m : k;
		int rowsB = tB ? n : k;
		int colsB = tB ? k : n;

		MatrixBlock A = MatrixBlock.randOperations(rowsA, colsA, 1.0, -1, 1, "uniform", 7);
		MatrixBlock B = MatrixBlock.randOperations(rowsB, colsB, 1.0, -1, 1, "uniform", 3);
		MatrixBlock C = new MatrixBlock(m, n, false);
		C.allocateDenseBlock();


		for(int i=0; i<50; i++) {
			runOldMethod(A, B, tA, tB);
			runNewKernel(A, B, C, tA, tB);
		}


		long startTimeOld = System.nanoTime();
		for(int i = 0; i < REP; i++) {
			runOldMethod(A, B, tA, tB);
		}
		double avgTimeOld = (System.nanoTime() - startTimeOld) / 1e6 / REP;


		double startTimeNew = System.nanoTime();
		for(int i = 0; i < REP; i++) {
			runNewKernel(A, B, C, tA, tB);
		}
		double avgTimeNew = (System.nanoTime() - startTimeNew) / 1e6 / REP;

		System.out.printf("Old Method: %.3f ms | New Kernel: %.3f ms", avgTimeOld, avgTimeNew);

		Assert.assertTrue(avgTimeNew < avgTimeOld);
	}

	private void runNewKernel(MatrixBlock A, MatrixBlock B, MatrixBlock C, boolean tA, boolean tB) {
		C.reset();

		LibMatrixMult.matrixMultDenseDenseMM(A.getDenseBlock(), B.getDenseBlock(), C.getDenseBlock(), tA, tB, m, k, 0, m, 0, n);
	}

	private void runOldMethod(MatrixBlock A, MatrixBlock B, boolean tA, boolean tB) {
		// do transpose if needed
		MatrixBlock A_in = tA ? LibMatrixReorg.transpose(A) : A;
		MatrixBlock B_in = tB ? LibMatrixReorg.transpose(B) : B;

		MatrixBlock C = new MatrixBlock(m, n, false);
		C.allocateDenseBlock();

		LibMatrixMult.matrixMultDenseDenseMM(A_in.getDenseBlock(), B_in.getDenseBlock(), C.getDenseBlock(), false,
				false, m, k, 0, m, 0, n);
	}
}
