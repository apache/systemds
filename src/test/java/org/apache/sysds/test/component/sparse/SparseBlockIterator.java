/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.component.sparse;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.sysds.runtime.data.SparseBlockFactory;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

/**
 * This is a component test for sparse matrix block, focusing on the iterator functionality for both general iteration
 * over non-zero cells and specific iteration over non-zero rows. To ensure comprehensive coverage, the tests encompass
 * full and partial iterators, different sparsity values, and explicitly verify the correct identification and iteration
 * over non-zero rows in the matrix.
 */
public class SparseBlockIterator extends AutomatedTestBase {
	private final static int rows = 324;
	private final static int cols = 100;
	private final static int rlVal = 134;
	private final static int ruVal = 253;
	private final static int clVal = 34;
	private final static int cuVal = 53;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.2;
	private final static double sparsity3 = 0.3;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testSparseBlockMCSR1Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity1, false, false);
	}

	@Test
	public void testSparseBlockMCSR2Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity2, false, false);
	}

	@Test
	public void testSparseBlockMCSR3Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity3, false, false);
	}

	@Test
	public void testSparseBlockMCSR1RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity1, true, false);
	}

	@Test
	public void testSparseBlockMCSR2RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity2, true, false);
	}

	@Test
	public void testSparseBlockMCSR3RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity3, true, false);
	}

	@Test
	public void testSparseBlockMCSR1RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity1, false, true);
	}

	@Test
	public void testSparseBlockMCSR2RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity2, false, true);
	}

	@Test
	public void testSparseBlockMCSR3RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity3, false, true);
	}

	@Test
	public void testSparseBlockMCSR1Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity1, true, true);
	}

	@Test
	public void testSparseBlockMCSR2Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity2, true, true);
	}

	@Test
	public void testSparseBlockMCSR3Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSR, sparsity3, true, true);
	}

	@Test
	public void testSparseBlockCSR1Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity1, false, false);
	}

	@Test
	public void testSparseBlockCSR2Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity2, false, false);
	}

	@Test
	public void testSparseBlockCSR3Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity3, false, false);
	}

	@Test
	public void testSparseBlockCSR1RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity1, true, false);
	}

	@Test
	public void testSparseBlockCSR2RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity2, true, false);
	}

	@Test
	public void testSparseBlockCSR3RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity3, true, false);
	}

	@Test
	public void testSparseBlockCSR1RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity1, false, true);
	}

	@Test
	public void testSparseBlockCSR2RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity2, false, true);
	}

	@Test
	public void testSparseBlockCSR3RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity3, false, true);
	}

	@Test
	public void testSparseBlockCSR1Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity1, true, true);
	}

	@Test
	public void testSparseBlockCSR2Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity2, true, true);
	}

	@Test
	public void testSparseBlockCSR3Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSR, sparsity3, true, true);
	}

	@Test
	public void testSparseBlockCOO1Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity1, false, false);
	}

	@Test
	public void testSparseBlockCOO2Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity2, false, false);
	}

	@Test
	public void testSparseBlockCOO3Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity3, false, false);
	}

	@Test
	public void testSparseBlockCOO1RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity1, true, false);
	}

	@Test
	public void testSparseBlockCOO2RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity2, true, false);
	}

	@Test
	public void testSparseBlockCOO3RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity3, true, false);
	}

	@Test
	public void testSparseBlockCOO1RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity1, false, true);
	}

	@Test
	public void testSparseBlockCOO2RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity2, false, true);
	}

	@Test
	public void testSparseBlockCOO3RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity3, false, true);
	}

	@Test
	public void testSparseBlockCOO1Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity1, true, true);
	}

	@Test
	public void testSparseBlockCOO2Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity2, true, true);
	}

	@Test
	public void testSparseBlockCOO3Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.COO, sparsity3, true, true);
	}

	@Test
	public void testSparseBlockDCSR1Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity1, false, false);
	}

	@Test
	public void testSparseBlockDCSR2Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity2, false, false);
	}

	@Test
	public void testSparseBlockDCSR3Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity3, false, false);
	}

	@Test
	public void testSparseBlockDCSR1RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity1, true, false);
	}

	@Test
	public void testSparseBlockDCSR2RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity2, true, false);
	}

	@Test
	public void testSparseBlockDCSR3RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity3, true, false);
	}

	@Test
	public void testSparseBlockDCSR1RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity1, false, true);
	}

	@Test
	public void testSparseBlockDCSR2RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity2, false, true);
	}

	@Test
	public void testSparseBlockDCSR3RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity3, false, true);
	}

	@Test
	public void testSparseBlockDCSR1Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity1, true, true);
	}

	@Test
	public void testSparseBlockDCSR2Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity2, true, true);
	}

	@Test
	public void testSparseBlockDCSR3Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.DCSR, sparsity3, true, true);
	}

	@Test
	public void testSparseBlockMCSC1Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity1, false, false);
	}

	@Test
	public void testSparseBlockMCSC2Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity2, false, false);
	}

	@Test
	public void testSparseBlockMCSC3Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity3, false, false);
	}

	@Test
	public void testSparseBlockMCSC1RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity1, true, false);
	}

	@Test
	public void testSparseBlockMCSC2RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity2, true, false);
	}

	@Test
	public void testSparseBlockMCSC3RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity3, true, false);
	}

	@Test
	public void testSparseBlockMCSC1RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity1, false, true);
	}

	@Test
	public void testSparseBlockMCSC2RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity2, false, true);
	}

	@Test
	public void testSparseBlockMCSC3RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity3, false, true);
	}

	@Test
	public void testSparseBlockMCSC1Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity1, true, true);
	}

	@Test
	public void testSparseBlockMCSC2Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity2, true, true);
	}

	@Test
	public void testSparseBlockMCSC3Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.MCSC, sparsity3, true, true);
	}

	@Test
	public void testSparseBlockCSC1Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity1, false, false);
	}

	@Test
	public void testSparseBlockCSC2Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity2, false, false);
	}

	@Test
	public void testSparseBlockCSC3Full() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity3, false, false);
	}

	@Test
	public void testSparseBlockCSC1RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity1, true, false);
	}

	@Test
	public void testSparseBlockCSC2RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity2, true, false);
	}

	@Test
	public void testSparseBlockCSC3RlPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity3, true, false);
	}

	@Test
	public void testSparseBlockCSC1RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity1, false, true);
	}

	@Test
	public void testSparseBlockCSC2RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity2, false, true);
	}

	@Test
	public void testSparseBlockCSC3RuPartial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity3, false, true);
	}

	@Test
	public void testSparseBlockCSC1Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity1, true, true);
	}

	@Test
	public void testSparseBlockCSC2Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity2, true, true);
	}

	@Test
	public void testSparseBlockCSC3Partial() {
		runSparseBlockIteratorTest(SparseBlock.Type.CSC, sparsity3, true, true);
	}



	private void runSparseBlockIteratorTest(SparseBlock.Type btype, double sparsity, boolean rlPartial, boolean ruPartial) {
		try {
			//data generation
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 8765432);

			//init sparse block
			MatrixBlock mbtmp = DataConverter.convertToMatrixBlock(A);
			SparseBlock srtmp = mbtmp.getSparseBlock();
			SparseBlock sblock = SparseBlockFactory.copySparseBlock(btype, srtmp, true);

			//check for correct number of non-zeros
			int[] rnnz = new int[rows];
			int nnz = 0;
			int rl = rlPartial ? rlVal : 0;
			int ru = ruPartial ? ruVal : rows;
			int cl = rlPartial && ruPartial ? clVal : 0;
			int cu = rlPartial && ruPartial ? cuVal : cols;
			for(int i = rl; i < ru; i++) {
				for(int j = cl; j < cu; j++)
					rnnz[i] += (A[i][j] != 0) ? 1 : 0;
				nnz += rnnz[i];
			}
			if(!rlPartial && !ruPartial && nnz != sblock.size()) // no restriction
				Assert.fail("Wrong number of non-zeros: " + sblock.size() + ", expected: " + nnz);

			//check correct isEmpty return
			if(!(rlPartial && ruPartial)) { // cols not restricted
				for(int i = rl; i < ru; i++)
					if(sblock.isEmpty(i) != (rnnz[i] == 0))
						Assert.fail("Wrong isEmpty(row) result for row nnz: " + rnnz[i]);
			}

			//check correct values	
			Iterator<IJV> iter = rlPartial && ruPartial ? sblock.getIterator(rl, ru, cl, cu): rlPartial? sblock.getIterator(rl, rows) : ruPartial? sblock.getIterator(ru) : sblock.getIterator();
			int count = 0;
			while(iter.hasNext()) {
				IJV cell = iter.next();
				if(cell.getI() < rl || cell.getI() >= ru)
					Assert.fail("iterator row outside of range");
				if(cell.getJ() < cl || cell.getJ() >= cu)
					Assert.fail("iterator column outside of range");
				if(cell.getV() != A[cell.getI()][cell.getJ()])
					Assert.fail("Wrong value returned by iterator: " + cell.getV() + ", expected: " +
						A[cell.getI()][cell.getJ()]);
				count++;
			}
			if(count != nnz)
				Assert.fail("Wrong number of values returned by iterator: " + count + ", expected: " + nnz);

			// check iterator over non-zero rows
			List<Integer> manualNonZeroRows = new ArrayList<>();
			List<Integer> iteratorNonZeroRows = new ArrayList<>();
			Iterator<Integer> iterRows = sblock.getNonEmptyRowsIterator(rl, ru);

			for(int i = rl; i < ru; i++)
				if(!sblock.isEmpty(i))
					manualNonZeroRows.add(i);
			while(iterRows.hasNext()) {
				iteratorNonZeroRows.add(iterRows.next());
			}

			// Compare the results
			if(!manualNonZeroRows.equals(iteratorNonZeroRows)) {
				Assert.fail("Verification of iterator over non-zero rows failed.");
			}

			// check second iterator over non-zero rows
			Iterator<Integer> iterRows2 = !rlPartial && !ruPartial? sblock.getNonEmptyRows().iterator() : sblock.getNonEmptyRows(rl, ru).iterator();
			List<Integer> iter2NonZeroRows = new ArrayList<>();

			while(iterRows2.hasNext()) {
				iter2NonZeroRows.add(iterRows2.next());
			}
			if(!manualNonZeroRows.equals(iter2NonZeroRows)) {
				Assert.fail("Verification of second iterator over non-zero rows failed.");
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
	}
}
