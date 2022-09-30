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
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class FederatedCoordinatorIntegrationCRUDTest extends FederatedMonitoringTestBase {
	private final static String TEST_NAME = "FederatedCoordinatorIntegrationCRUDTest";

	private final static String TEST_DIR = "functions/federated/monitoring/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedCoordinatorIntegrationCRUDTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
		startFedMonitoring(null);
	}

	@Test
	public void testCoordinatorAddedForMonitoring() {
		var addedCoordinators = addEntities(1, Entity.COORDINATOR);
		var firstCoordinatorStatus = addedCoordinators.get(0).statusCode();

		Assert.assertEquals("Added coordinator status code", HttpStatus.SC_OK, firstCoordinatorStatus);
	}

	@Test
	public void testCoordinatorRemovedFromMonitoring() {
		addEntities(2, Entity.COORDINATOR);
		var statusCode = removeEntity(1L, Entity.COORDINATOR).statusCode();

		var getAllCoordinatorsResponse = getEntities(Entity.COORDINATOR);
		var numReturnedCoordinators = StringUtils.countMatches(getAllCoordinatorsResponse.body().toString(), "id");

		Assert.assertEquals("Removed coordinator status code", HttpStatus.SC_OK, statusCode);
		Assert.assertEquals("Removed coordinators num", 1, numReturnedCoordinators);
	}

	@Test
	public void testCoordinatorDataUpdated() {
		addEntities(3, Entity.COORDINATOR);
		var newCoordinatorData = new CoordinatorModel(1L, "NonExistentName", "nonexistent.address", 4241L);

		var editedCoordinator = updateEntity(newCoordinatorData, Entity.COORDINATOR);

		var getAllCoordinatorsResponse = getEntities(Entity.COORDINATOR);
		var numCoordinatorsNewData = StringUtils.countMatches(getAllCoordinatorsResponse.body().toString(), newCoordinatorData.name);

		Assert.assertEquals("Updated coordinator status code", HttpStatus.SC_OK, editedCoordinator.statusCode());
		Assert.assertEquals("Updated coordinator name", 1, numCoordinatorsNewData);
	}

	@Test
	public void testCorrectAmountAddedCoordinatorsForMonitoring() {
		int numCoordinators = 3;
		var addedCoordinators = addEntities(numCoordinators, Entity.COORDINATOR);

		for (int i = 0; i < numCoordinators; i++) {
			var coordinatorStatus = addedCoordinators.get(i).statusCode();
			Assert.assertEquals("Added coordinator status code", HttpStatus.SC_OK, coordinatorStatus);
		}

		var getAllCoordinatorsResponse = getEntities(Entity.COORDINATOR);
		var numReturnedCoordinators = StringUtils.countMatches(getAllCoordinatorsResponse.body().toString(), "id");

		Assert.assertEquals("Amount of coordinators to get", numCoordinators, numReturnedCoordinators);
	}
}
