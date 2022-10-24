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

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.Collection;

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.cost.InstructionTypeCounter;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FedWorkerMatrixMultiplyWorkload extends FedWorkerBase {

	private static final String confC = "src/test/resources/component/federated/workload.xml";

	private final MatrixBlock mbl;
	private final MatrixBlock mbr;

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int port = startWorker(confC);

		final MatrixBlock L_mb1x1000 = TestUtils.generateTestMatrixBlock(1, 100, 0.5, 9.5, 1.0, 1342);
		final MatrixBlock R_mb1000x10 = TestUtils.generateTestMatrixBlock(100, 100, 0.5, 2.5, 1.0, 222);
		final MatrixBlock L_mb1x1000_r = TestUtils.round(L_mb1x1000);
		final MatrixBlock R_mb1000x10_r = TestUtils.round(R_mb1000x10);

		tests.add(new Object[] {port, L_mb1x1000, R_mb1000x10_r});
		tests.add(new Object[] {port, L_mb1x1000_r, R_mb1000x10_r});

		return tests;
	}

	public FedWorkerMatrixMultiplyWorkload(int port, MatrixBlock mbl, MatrixBlock mbr) {
		super(port);
		this.mbl = mbl;
		this.mbr = mbr;
	}

	@Test
	public void verifySameOrAlsoCompressedAsLocalCompress() {
		// Local
		final InstructionTypeCounter c =  new InstructionTypeCounter(0, 0, 0, 1000, 0, 0, 0, 0, false);
		final MatrixBlock mbcLocal = CompressedMatrixBlockFactory.compress(mbr, c).getLeft();
		if(!(mbcLocal instanceof CompressedMatrixBlock))
			return; // would not compress anyway so skip
		
		// Local multiply once
		final MatrixBlock e1 = LibMatrixMult.matrixMult(mbl, mbr);
		if(e1.getNumColumns() != mbr.getNumRows()) {
			LOG.error(e1.getNumColumns() + " " + mbr.getNumRows());
			return; // skipping because test is invalid
		}

		// Federated
		final long idl = putMatrixBlock(mbl);
		final long idr = putMatrixBlock(mbr);
		long ide = matrixMult(idl, idr);
		for(int i = 0; i < 9; i++) // chain left side compressed multiplications with idr.
			ide = matrixMult(ide, idr);

		// give the federated site time to compress async (it should already be done, but just to be safe).
		FederatedTestUtils.wait(1000);

		// Get back the matrix block stored behind mbr that should be compressed now.
		final MatrixBlock mbr_compressed = getMatrixBlock(idr);

		if(!(mbr_compressed instanceof CompressedMatrixBlock))
			fail("Invalid result, the federated site did not compress the matrix block based on workload");

		TestUtils.compareMatricesBitAvgDistance(mbcLocal, mbr_compressed, 0, 0,
			"Not equivalent matrix block returned from federated site");
	}



}
