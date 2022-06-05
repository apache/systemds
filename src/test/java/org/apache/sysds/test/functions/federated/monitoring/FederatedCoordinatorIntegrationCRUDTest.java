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
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.NodeEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.EntityEnum;
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
		var addedCoordinators = addEntities(EntityEnum.COORDINATOR,1);
		var firstCoordinatorStatus = addedCoordinators.get(0).statusCode();

		Assert.assertEquals("Added coordinator status code", HttpStatus.SC_OK, firstCoordinatorStatus);
	}

	@Test
	@Ignore
	public void testCoordinatorRemovedFromMonitoring() {
		addEntities(EntityEnum.COORDINATOR,2);
		var statusCode = removeEntity(EntityEnum.COORDINATOR,1L).statusCode();

		var getAllCoordinatorsResponse = getEntities(EntityEnum.COORDINATOR);
		var numReturnedCoordinators = StringUtils.countMatches(getAllCoordinatorsResponse.body().toString(), "id");

		Assert.assertEquals("Removed coordinator status code", HttpStatus.SC_OK, statusCode);
		Assert.assertEquals("Removed coordinators num", 1, numReturnedCoordinators);
	}

	@Test
	@Ignore
	public void testCoordinatorDataUpdated() {
		addEntities(EntityEnum.COORDINATOR,3);
		var newCoordinatorData = new NodeEntityModel(1L, "NonExistentName", "nonexistent.address");

		var editedCoordinator = updateEntity(EntityEnum.COORDINATOR, newCoordinatorData);

		var getAllCoordinatorsResponse = getEntities(EntityEnum.COORDINATOR);
		var numCoordinatorsNewData = StringUtils.countMatches(getAllCoordinatorsResponse.body().toString(), newCoordinatorData.getName());

		Assert.assertEquals("Updated coordinator status code", HttpStatus.SC_OK, editedCoordinator.statusCode());
		Assert.assertEquals("Updated coordinators num", 1, numCoordinatorsNewData);
	}

	@Test
	@Ignore
	public void testCorrectAmountAddedCoordinatorsForMonitoring() {
		int numCoordinators = 3;
		var addedCoordinators = addEntities(EntityEnum.COORDINATOR, numCoordinators);

		for (int i = 0; i < numCoordinators; i++) {
			var coordinatorStatus = addedCoordinators.get(i).statusCode();
			Assert.assertEquals("Added coordinator status code", HttpStatus.SC_OK, coordinatorStatus);
		}

		var getAllCoordinatorsResponse = getEntities(EntityEnum.COORDINATOR);
		var numReturnedCoordinators = StringUtils.countMatches(getAllCoordinatorsResponse.body().toString(), "id");

		Assert.assertEquals("Amount of coordinators to get", numCoordinators, numReturnedCoordinators);
	}
}
