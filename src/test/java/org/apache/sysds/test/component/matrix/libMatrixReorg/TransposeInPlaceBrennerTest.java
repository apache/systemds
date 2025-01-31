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

package org.apache.sysds.test.component.matrix.libMatrixReorg;

import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

public class TransposeInPlaceBrennerTest {

	@Test
	public void transposeInPlaceDenseBrennerOnePrime() {
		// 3*4-1 = 11
		testTransposeInPlaceDense(3, 4, 1);
	}

	@Test
	public void transposeInPlaceDenseBrennerTwoPrimes() {
		// 4*9-1 = 5*7
		testTransposeInPlaceDense(4, 9, 0.96);
	}

	@Test
	public void transposeInPlaceDenseBrennerThreePrimes() {
		// 2*53-1 = 3*5*7
		testTransposeInPlaceDense(2, 53, 0.52);
	}

	@Test
	public void transposeInPlaceDenseBrennerThreePrimesOneExpo() {
		// 1151*2999-1 = (2**3)*3*143827
		testTransposeInPlaceDense(1151, 2999, 0.82);
	}

	@Test
	public void transposeInPlaceDenseBrennerThreePrimesAllExpos() {
		// 9*10889-1 = (2**4)*(5**3)*(7**2)
		testTransposeInPlaceDense(9, 10889, 0.74);
	}

	@Test
	public void transposeInPlaceDenseBrennerFourPrimesOneExpo() {
		// 53*4421-1 = (2**3)*3*13*751
		testTransposeInPlaceDense(53, 4421, 0.75);
	}

	@Test
	public void transposeInPlaceDenseBrennerFivePrimes() {
		// 3*3337-1 = 2*5*7*11*13
		testTransposeInPlaceDense(3, 3337, 0.68);
	}

	@Test
	public void transposeInPlaceDenseBrennerSixPrimesOneExpo() {
		// 53*7177-1 = (2**2)*5*7*11*13*19
		testTransposeInPlaceDense(53, 7177, 0.78);
	}

	@Test
	public void transposeInPlaceDenseBrennerSevenPrimesThreeExpos() {
		// 2087*17123-1 = (2**2)*3*(5**2)*(7**2)*11*13*17
		testTransposeInPlaceDense(2087, 17123, 0.79);
	}

	@Test
	public void transposeInPlaceDenseBrennerEightPrimes() {
		// 347*27953-1 = 2*3*5*7*11*13*17*19
		testTransposeInPlaceDense(347, 27953, 0.86);
	}

	@Test
	public void transposeInPlaceDenseBrennerNinePrimes() {
		// 317*703763-1 = 2*3*5*7*11*13*17*19*23
		MatrixBlock X = MatrixBlock.randOperations(317, 703763, 0.52);

		Exception exception = assertThrows(RuntimeException.class,
			() -> LibMatrixReorg.transposeInPlaceDenseBrenner(X, 1));
		assertTrue(exception.getMessage().contains("Not enough space, need to expand input arrays."));
	}

	private void testTransposeInPlaceDense(int rows, int cols, double sparsity) {
		MatrixBlock X = MatrixBlock.randOperations(rows, cols, sparsity);
		MatrixBlock tX = LibMatrixReorg.transpose(X);

		LibMatrixReorg.transposeInPlaceDenseBrenner(X, 1);

		TestUtils.compareMatrices(X, tX, 0);
	}

}
