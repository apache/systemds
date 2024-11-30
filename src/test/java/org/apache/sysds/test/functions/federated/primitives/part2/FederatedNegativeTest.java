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

package org.apache.sysds.test.functions.federated.primitives.part2;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class FederatedNegativeTest {
	@Test
	public void NegativeTest1() {
		int port = AutomatedTestBase.getRandomAvailablePort();
		Thread t = null;
		try{
			String[] args = {"-w", Integer.toString(port)};
			t = AutomatedTestBase.startLocalFedWorkerWithArgs(args);
			Thread.sleep(2000);
		} catch(Exception e){
			NegativeTest1();
		}
		FederationUtils.resetFedDataID(); //ensure expected ID when tests run in single JVM
		List<Pair<FederatedRange, FederatedData>> fedMap = new ArrayList<>();
		FederatedRange r = new FederatedRange(new long[]{0,0}, new long[]{1,1});
		FederatedData d = new FederatedData(Types.DataType.SCALAR,
			new InetSocketAddress("localhost", port), "Nowhere");
		fedMap.add(Pair.of(r,d));
		FederationMap fedM = new FederationMap(fedMap);
		FederatedRequest fr = new FederatedRequest(FederatedRequest.RequestType.GET_VAR);
		Future<FederatedResponse>[] res = fedM.execute(0, fr);
		try {
			FederatedResponse fres = res[0].get();
			assertFalse(fres.isSuccessful());
			assertTrue(fres.getErrorMessage().contains("Variable 0 does not exist at federated worker"));
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		finally {
			//robustness in single JVM tests
			FederatedData.resetFederatedSites();
		}
		TestUtils.shutdownThread(t);
	}
}
