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

package org.apache.sysds.test.functions.federated.monitoring;

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.WorkerService;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FederatedWorkerStatisticsTest extends FederatedMonitoringTestBase {
	private final static String TEST_NAME = "FederatedWorkerStatisticsTest";

	private final static String TEST_DIR = "functions/federated/monitoring/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedWorkerStatisticsTest.class.getSimpleName() + "/";

	private static int[] workerPorts;
	private final WorkerService workerMonitoringService = new WorkerService();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
		workerPorts = startFedWorkers(3);
	}

	@Test
	public void testWorkerStatisticsReturnedForMonitoring() {
		workerMonitoringService.create(new BaseEntityModel(1L, "Worker", "localhost:" + workerPorts[0]));

		var model = workerMonitoringService.get(1L);
		var modelData = model.getData();

		Assert.assertNotNull("Data field of model contains worker statistics", model.getData());
		Assert.assertNotEquals("Data field of model contains worker statistics",0, modelData.length());
		Assert.assertTrue("Data field of model contains worker statistics", modelData.contains("JVM"));
	}
}
