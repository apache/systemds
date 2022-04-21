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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FedWorkerMatrix extends FedWorkerBase {

	private final MatrixBlock mb;
	private final int rep;

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int port = startWorker();

		final MatrixBlock mb10x10 = TestUtils.generateTestMatrixBlock(10, 10, 0.5, 9.5, 1.0, 1342);

		tests.add(new Object[] {port, mb10x10, 10});
		tests.add(new Object[] {port, TestUtils.round(mb10x10), 10});

		final MatrixBlock mb1000x10 = TestUtils.generateTestMatrixBlock(1000, 10, 0.5, 9.5, 1.0, 1342);
		tests.add(new Object[] {port, mb1000x10, 10});

		final MatrixBlock mb10x1000 = TestUtils.generateTestMatrixBlock(10, 1000, 0.5, 9.5, 1.0, 1342);
		tests.add(new Object[] {port, mb10x1000, 10});

		return tests;
	}

	public FedWorkerMatrix(int port, MatrixBlock mb, int rep) {
		super(port);
		this.mb = mb;
		this.rep = rep;
	}

	@Test
	public void verifyPutGetMatrixBlock() {
		for(int i = 0; i < rep; i++) {
			final long id = putMatrixBlock(mb);
			final MatrixBlock mbr = getMatrixBlock(id);
			TestUtils.compareMatricesBitAvgDistance(mb, mbr, 0, 0,
				"Not equivalent matrix block returned from federated site");
		}
	}

	@Test
	public void verifyPutOnceGetSameMatrixBlock() {
		final long id = putMatrixBlock(mb);
		final MatrixBlock mbrInit = getMatrixBlock(id);
		for(int i = 0; i < rep; i++) {
			final MatrixBlock mbr = getMatrixBlock(id);
			TestUtils.compareMatricesBitAvgDistance(mbrInit, mbr, 0, 0,
				"Not equivalent matrix block returned from federated site");
		}
	}

}
