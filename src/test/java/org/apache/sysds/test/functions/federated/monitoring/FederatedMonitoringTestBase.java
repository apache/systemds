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

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.test.functions.federated.multitenant.MultiTenantTestBase;
import org.junit.After;

import com.fasterxml.jackson.databind.ObjectMapper;

@net.jcip.annotations.NotThreadSafe
public abstract class FederatedMonitoringTestBase extends MultiTenantTestBase {
	protected Process monitoringProcess;
	private int monitoringPort;

	private static final String MAIN_URI = "http://localhost";

	private static final String WORKER_MAIN_PATH = "/workers";
	private static final String COORDINATOR_MAIN_PATH = "/coordinators";
	private static final String STATISTICS_MAIN_PATH = "/statistics";

	public enum Entity {
		WORKER,
		COORDINATOR,
		STATISTICS
	}

	@Override
	public abstract void setUp();

	// ensure that the processes are killed - even if the test throws an exception
	@After
	public void stopMonitoringProcesses() {
		if (monitoringProcess != null) {
			monitoringProcess.destroyForcibly();
		}
	}

	/**
	 * Start federated backend monitoring processes on available port
	 *
	 * @return
	 */
	protected void startFedMonitoring(String[] addArgs) {
		monitoringPort = getRandomAvailablePort();
		monitoringProcess = startLocalFedMonitoring(monitoringPort, addArgs);
	}

	protected List<HttpResponse<?>> addEntities(int count, Entity entity) {
		String uriStr = MAIN_URI + ":" + monitoringPort + WORKER_MAIN_PATH;
		String name = "Worker";

		if (entity == Entity.COORDINATOR) {
			uriStr = MAIN_URI + ":" + monitoringPort + COORDINATOR_MAIN_PATH;
			name = "Coordinator";
		}

		List<HttpResponse<?>> responses = new ArrayList<>();
		try {
			ObjectMapper objectMapper = new ObjectMapper();
			for (int i = 0; i < count; i++) {
				String requestBody = "";

				if (entity == Entity.WORKER) {
					requestBody = objectMapper
							.writerWithDefaultPrettyPrinter()
							.writeValueAsString(new WorkerModel((i + 1L), name, "localhost"));
				} else if (entity == Entity.COORDINATOR) {
					CoordinatorModel model = new CoordinatorModel(i + 1L, name, "localhost", 4242L);
					model.generateMonitoringKey();

					requestBody = objectMapper
							.writerWithDefaultPrettyPrinter()
							.writeValueAsString(model);
				}
				var client = HttpClient.newHttpClient();
				var request = HttpRequest.newBuilder(URI.create(uriStr))
					.header("accept", "application/json")
					.POST(HttpRequest.BodyPublishers.ofString(requestBody))
					.build();
				responses.add(client.send(request, HttpResponse.BodyHandlers.ofString()));
			}

			return responses;
		}
		catch (IOException | InterruptedException e) {
			throw new RuntimeException(e);
		}
	}

	protected HttpResponse<?> updateEntity(BaseModel editModel, Entity entity) {
		String uriStr = MAIN_URI + ":" + monitoringPort + WORKER_MAIN_PATH + "/" + editModel.id;

		if (entity == Entity.COORDINATOR) {
			uriStr = MAIN_URI + ":" + monitoringPort + COORDINATOR_MAIN_PATH + "/" + editModel.id;
		}

		try {
			ObjectMapper objectMapper = new ObjectMapper();
			String requestBody = "";

			if (entity == Entity.WORKER) {
				requestBody = objectMapper
						.writerWithDefaultPrettyPrinter()
						.writeValueAsString(new WorkerModel(editModel.id, ((WorkerModel) editModel).name, ((WorkerModel) editModel).address));
			} else if (entity == Entity.COORDINATOR) {
				CoordinatorModel model = new CoordinatorModel(
						editModel.id, ((CoordinatorModel)editModel).name,((CoordinatorModel)editModel).host, ((CoordinatorModel)editModel).processId);
				model.generateMonitoringKey();

				requestBody = objectMapper
					.writerWithDefaultPrettyPrinter()
					.writeValueAsString(model);
			}
			var client = HttpClient.newHttpClient();
			var request = HttpRequest.newBuilder(URI.create(uriStr))
				.header("accept", "application/json")
				.PUT(HttpRequest.BodyPublishers.ofString(requestBody))
				.build();

			return client.send(request, HttpResponse.BodyHandlers.ofString());
		}
		catch (IOException | InterruptedException e) {
			throw new RuntimeException(e);
		}
	}

	protected HttpResponse<?> removeEntity(Long id, Entity entity) {
		String uriStr = MAIN_URI + ":" + monitoringPort + WORKER_MAIN_PATH + "/" + id;

		if (entity == Entity.COORDINATOR) {
			uriStr = MAIN_URI + ":" + monitoringPort + COORDINATOR_MAIN_PATH + "/" + id;
		}

		try {
			var client = HttpClient.newHttpClient();
			var request = HttpRequest.newBuilder(URI.create(uriStr))
				.header("accept", "application/json")
				.DELETE()
				.build();

			return client.send(request, HttpResponse.BodyHandlers.ofString());
		}
		catch (IOException | InterruptedException e) {
			throw new RuntimeException(e);
		}
	}

	protected HttpResponse<?> getEntities(Entity entity) {
		String uriStr = MAIN_URI + ":" + monitoringPort + WORKER_MAIN_PATH;

		if (entity == Entity.COORDINATOR) {
			uriStr = MAIN_URI + ":" + monitoringPort + COORDINATOR_MAIN_PATH;
		}

		if (entity == Entity.STATISTICS) {
			uriStr = MAIN_URI + ":" + monitoringPort + STATISTICS_MAIN_PATH + "/1";
		}

		try {
			var client = HttpClient.newHttpClient();
			var request = HttpRequest.newBuilder(URI.create(uriStr))
				.header("accept", "application/json")
				.GET().build();
			return client.send(request, HttpResponse.BodyHandlers.ofString());
		}
		catch (IOException | InterruptedException e) {
			throw new RuntimeException(e);
		}
	}
}
