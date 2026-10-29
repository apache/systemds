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

package org.apache.sysds.test.functions.federated.io;

import java.net.InetSocketAddress;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

@Ignore("Heavy test: Transfer a matrix of size greater than 2GB over network. "
	+ "This test needs ~9GB heap, since client and worker share one JVM; to run "
	+ "this test, set the 'argLine' property inside the pom to '-Xmx9g'.")
public class FederatedMaxPayloadTest extends AutomatedTestBase {

	private final static String TEST_NAME = "FederatedMaxPayloadTest";
	private final static String TEST_DIR = "functions/federated/network/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedMaxPayloadTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {""}));
	}

	@Test
	public void transferOverTwoGigabytePayload() {
		int port = getRandomAvailablePort();
		startLocalFedWorkerThread(port);
		InetSocketAddress address = new InetSocketAddress("localhost", port);
		MatrixBlock mb = denseMatrixExceedingTwoGigabytes();
		try {
			FederatedRequest request = new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, 1, mb);

			Future<FederatedResponse> response = FederatedData.executeFederatedOperation(address, request);
			Assert.assertTrue("Network send was not successful.", response.get().isSuccessful());
		}
		catch(Exception e) {
			Assert.fail("Federated transfer failed: " + e.getMessage());
		}
		finally {
			FederatedData.clearFederatedWorkers();
		}
	}

	private static MatrixBlock denseMatrixExceedingTwoGigabytes() {
		int rows = 30000;
		int cols = 8950;
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		mb.allocateDenseBlock();
		mb.setNonZeros((long) rows * cols);
		return mb;
	}
}
