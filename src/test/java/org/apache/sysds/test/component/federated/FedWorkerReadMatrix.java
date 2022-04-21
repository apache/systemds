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

import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.ReaderTextCSV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

@RunWith(value = Parameterized.class)
public class FedWorkerReadMatrix extends FedWorkerBase {

	protected final String path;

	@Parameters
	public static Collection<Object[]> data() {
		final ArrayList<Object[]> tests = new ArrayList<>();

		final int port = startWorker();

		final String mb100x10 = "src/test/resources/component/federated/100x10.csv";
		final String mb1000x10 = "src/test/resources/component/federated/1000x10.csv";

		tests.add(new Object[] {port, mb100x10});
		tests.add(new Object[] {port, mb1000x10});

		return tests;
	}

	public FedWorkerReadMatrix(int port, String path) {
		super(port);
		this.path = path;
	}

	@Test
	public void verifyRead() {
		MatrixBlock expected = readCSV();
		Long id = readMatrix(path);
		MatrixBlock actual = getMatrixBlock(id);
		TestUtils.compareMatricesBitAvgDistance(expected, actual, 0, 0,
				"Not equivalent matrix block read from federated site");

	}

	protected MatrixBlock readCSV() {
		try {
			ReaderTextCSV reader = new ReaderTextCSV(new FileFormatPropertiesCSV());
			return reader.readMatrixFromHDFS(path, -1, -1, 1000, -1);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("Failed to read csv");
			return null;
		}
	}
}
