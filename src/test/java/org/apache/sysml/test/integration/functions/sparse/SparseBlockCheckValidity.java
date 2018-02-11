package org.apache.sysml.test.integration.functions.sparse;

import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlockCSR;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class SparseBlockCheckValidity extends AutomatedTestBase
{
	private final static int rows = 632;
	private final static int cols = 454;
	private final static double sparsity1 = 0.11;
	private final static double sparsity2 = 0.21;
	private final static double sparsity3 = 0.31;

	private enum InitType {
		SEQ_SET,
		RAND_SET,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockCSR1Seq() {
		runSparseBlockValidityTest(SparseBlock.Type.CSR, sparsity1, InitType.SEQ_SET);
	}

	private void runSparseBlockValidityTest( SparseBlock.Type btype, double sparsity, InitType itype) {
		{
			try {
				//data generation

				double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 7654321);

				//init sparse block
				SparseBlock sblock = null;
				switch (btype) {
					case CSR:
						sblock = new SparseBlockCSR(rows, cols);
						break;
				}

				if (itype == InitType.SEQ_SET) {
					for (int i = 0; i < rows; i++)
						for (int j = 0; j < cols; j++)
							sblock.append(i, j, A[i][j]);
				}

				//check for correct number of non-zeroes
				int[] rnnz = new int[rows];
				int nnz = 0;
				for (int i = 0; i < rows; i++) {
					for (int j = 0; j < cols; j++)
						rnnz[i] += (A[i][j] != 0) ? 1 : 0;
					nnz += rnnz[i];
				}

				//check correct values
				for (int i = 0; i < rows; i++)
					if (!sblock.isEmpty(i))
						for (int j = 0; j < cols; j++) {
							double tmp = sblock.get(i, j);
							if (tmp != A[i][j])
								Assert.fail("Wrong get value for cell (" + i + "," + j + "): " + tmp + ", expected: " + A[i][j]);
						}


			}
			catch(Exception ex) {
				ex.printStackTrace();
				throw new RuntimeException(ex);
			}
		}
	}

}
