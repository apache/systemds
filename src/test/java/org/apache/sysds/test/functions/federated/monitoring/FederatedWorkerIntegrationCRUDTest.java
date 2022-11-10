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

import org.apache.commons.lang.StringUtils;
import org.apache.http.HttpStatus;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class FederatedWorkerIntegrationCRUDTest extends FederatedMonitoringTestBase {
	private final static String TEST_NAME = "FederatedWorkerIntegrationCRUDTest";

	private final static String TEST_DIR = "functions/federated/monitoring/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedWorkerIntegrationCRUDTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
		startFedMonitoring(null);
	}

	@Test
	public void testWorkerAddedForMonitoring() {
		var addedWorkers = addEntities(1, Entity.WORKER);
		var firstWorkerStatus = addedWorkers.get(0).statusCode();

		Assert.assertEquals("Added worker status code", HttpStatus.SC_OK, firstWorkerStatus);
	}

	@Test
	public void testWorkerRemovedFromMonitoring() {
		addEntities(2, Entity.WORKER);
		var statusCode = removeEntity(1L, Entity.WORKER).statusCode();

		var getAllWorkersResponse = getEntities(Entity.WORKER);
		var numReturnedWorkers = StringUtils.countMatches(getAllWorkersResponse.body().toString(), "id");

		Assert.assertEquals("Removed worker status code", HttpStatus.SC_OK, statusCode);
		Assert.assertEquals("Removed workers num", 1, numReturnedWorkers);
	}

	@Test
	public void testWorkerDataUpdated() {
		addEntities(3, Entity.WORKER);
		var newWorkerData = new WorkerModel(1L, "NonExistentName", "nonexistent.address");

		var editedWorker = updateEntity(newWorkerData, Entity.WORKER);

		var getAllWorkersResponse = getEntities(Entity.WORKER);
		var numWorkersNewData = StringUtils.countMatches(getAllWorkersResponse.body().toString(), newWorkerData.name);

		Assert.assertEquals("Updated worker status code", HttpStatus.SC_OK, editedWorker.statusCode());
		Assert.assertEquals("Updated workers num", 1, numWorkersNewData);
	}

	@Test
	public void testCorrectAmountAddedWorkersForMonitoring() {
		int numWorkers = 3;
		var addedWorkers = addEntities(numWorkers, Entity.WORKER);

		for (int i = 0; i < numWorkers; i++) {
			var workerStatus = addedWorkers.get(i).statusCode();
			Assert.assertEquals("Added worker status code", HttpStatus.SC_OK, workerStatus);
		}

		var getAllWorkersResponse = getEntities(Entity.WORKER);
		var numReturnedWorkers = StringUtils.countMatches(getAllWorkersResponse.body().toString(), "id");

		Assert.assertEquals("Amount of workers to get", numWorkers, numReturnedWorkers);
	}
}
