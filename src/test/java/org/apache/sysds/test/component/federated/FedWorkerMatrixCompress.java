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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FedWorkerMatrixCompress extends FedWorkerBase {

	private static final String confC = "src/test/resources/component/federated/comp.xml";

	private final MatrixBlock mb;

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int port = startWorker(confC);

		final MatrixBlock mb1000x10 = TestUtils.generateTestMatrixBlock(1000, 10, 0.5, 2.5, 1.0, 1342);
		final MatrixBlock mb1000x10_r = TestUtils.round(mb1000x10);

		tests.add(new Object[] {port, mb1000x10}); // do not compress
		tests.add(new Object[] {port, mb1000x10_r}); // compress

		return tests;
	}

	public FedWorkerMatrixCompress(int port, MatrixBlock mb) {
		super(port);
		this.mb = mb;
	}

	@Test
	public void verifySameOrAlsoCompressedAsLocalCompress() {
		// local
		final MatrixBlock mbcLocal = CompressedMatrixBlockFactory.compress(mb).getLeft();

		// federated
		final long id = putMatrixBlock(mb);
		// give the federated site time to compress async.
		FederatedTestUtils.wait(1000);
		final MatrixBlock mbr = getMatrixBlock(id);

		if(mbcLocal instanceof CompressedMatrixBlock && !(mbr instanceof CompressedMatrixBlock))
			fail("Invalid result, the federated site did not compress the matrix block");

		TestUtils.compareMatricesBitAvgDistance(mbcLocal, mbr, 0, 0,
			"Not equivalent matrix block returned from federated site");
	}

}
