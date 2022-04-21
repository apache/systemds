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

package org.apache.sysds.test.component.federated;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FedWorkerMatrixMultiply extends FedWorkerBase {

	private final MatrixBlock mbl;
	private final MatrixBlock mbr;

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int port = startWorker();

		final MatrixBlock mb10x10_1 = TestUtils.generateTestMatrixBlock(10, 10, 0.5, 9.5, 1.0, 1342);
		final MatrixBlock mb10x10_2 = TestUtils.generateTestMatrixBlock(10, 10, 0.5, 9.5, 1.0, 222);
		final MatrixBlock mb10x10_1_r = TestUtils.round(mb10x10_1);
		final MatrixBlock mb10x10_2_r = TestUtils.round(mb10x10_2);

		tests.add(new Object[] {port, mb10x10_1, mb10x10_2});
		tests.add(new Object[] {port, mb10x10_1_r, mb10x10_2_r});

		final MatrixBlock mb3x10 = TestUtils.generateTestMatrixBlock(3, 10, 0.5, 9.5, 1.0, 324);
		final MatrixBlock mb10x4 = TestUtils.generateTestMatrixBlock(10, 4, 0.5, 9.5, 1.0, 324);

		tests.add(new Object[] {port, mb3x10, mb10x10_2});
		tests.add(new Object[] {port, mb10x10_1_r, mb10x4});

		return tests;
	}

	public FedWorkerMatrixMultiply(int port, MatrixBlock mbl, MatrixBlock mbr) {
		super(port);
		this.mbl = mbl;
		this.mbr = mbr;
	}

	@Test
	public void matrixMultiplication() {

		// local
		final MatrixBlock expected = LibMatrixMult.matrixMult(mbl, mbr);

		// Federated
		final long idl = putMatrixBlock(mbl);
		final long idr = putMatrixBlock(mbr);
		final long idOut = matrixMult(idl, idr);
		final MatrixBlock mbr = getMatrixBlock(idOut);

		// Compare
		TestUtils.compareMatricesBitAvgDistance(expected, mbr, 0, 0,
			"Not equivalent matrix block returned from federated site");
	}

	@Test
	public void matrixMultiplicationChainRight() {
		// local
		final MatrixBlock e1 = LibMatrixMult.matrixMult(mbl, mbr);
		if(e1.getNumColumns() != mbr.getNumRows())
			return; // skipping because test is invalid.

		final MatrixBlock e2 = LibMatrixMult.matrixMult(e1, mbr);
		final MatrixBlock e3 = LibMatrixMult.matrixMult(e2, mbr);

		// Federated
		final long idl = putMatrixBlock(mbl);
		final long idr = putMatrixBlock(mbr);
		final long ide1 = matrixMult(idl, idr);
		final long ide2 = matrixMult(ide1, idr);
		final long ide3 = matrixMult(ide2, idr);
		final MatrixBlock mbr = getMatrixBlock(ide3);

		// Compare
		TestUtils.compareMatricesBitAvgDistance(e3, mbr, 0, 0,
			"Not equivalent matrix block returned from federated site");
	}

	@Test
	public void matrixMultiplicationChainLeft() {
		// local
		final MatrixBlock e1 = LibMatrixMult.matrixMult(mbl, mbr);
		if(mbl.getNumColumns() != e1.getNumRows())
			return; // skipping because test is invalid.

		final MatrixBlock e2 = LibMatrixMult.matrixMult(mbl, e1);
		final MatrixBlock e3 = LibMatrixMult.matrixMult(mbl, e2);

		// Federated
		final long idl = putMatrixBlock(mbl);
		final long idr = putMatrixBlock(mbr);
		final long ide1 = matrixMult(idl, idr);
		final long ide2 = matrixMult(idl, ide1);
		final long ide3 = matrixMult(idl, ide2);
		final MatrixBlock mbr = getMatrixBlock(ide3);

		// Compare
		TestUtils.compareMatricesBitAvgDistance(e3, mbr, 0, 0,
			"Not equivalent matrix block returned from federated site");
	}

}
