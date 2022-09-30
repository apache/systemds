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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.DataObjectModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventStageModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.RequestModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsOptions;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.Constants;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.StatisticsService;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.WorkerService;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class FederatedWorkerStatisticsTest extends FederatedMonitoringTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedWorkerStatisticsTest.class.getName());

	private final static String TEST_NAME = "FederatedWorkerStatisticsTest";

	private final static String TEST_DIR = "functions/federated/monitoring/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedWorkerStatisticsTest.class.getSimpleName() + "/";

	private static final String PERFORMANCE_FORMAT = "For %d number of workers, milliseconds elapsed %d.";

	private static int[] workerPorts;
	private final IRepository entityRepository = new DerbyRepository();
	private final WorkerService workerMonitoringService = new WorkerService();
	private final StatisticsService statisticsMonitoringService = new StatisticsService();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
		workerPorts = startFedWorkers(6);
	}

	@Test
	public void testWorkerStatisticsParsedCorrectly() {

		var model = (StatisticsModel) StatisticsService.getWorkerStatistics(1L, "localhost:" + workerPorts[0]);

		Assert.assertNotNull("Stats parsed correctly", model);
		Assert.assertNotEquals("Utilization stats parsed correctly", 0, model.utilization.size());
	}

	@Test
	@Ignore
	public void testWorkerStatisticsPerformance() throws InterruptedException {
		ExecutorService executor = Executors.newFixedThreadPool(workerPorts.length);

		double meanExecTime = 0.f;
		double numRepetitionsExperiment = 100.f;

		for (int j = -10; j < numRepetitionsExperiment; j++) {

			Collection<Callable<StatisticsModel>> collect = new ArrayList<>();
			Collection<Callable<Boolean>> parse = new ArrayList<>();

			for (int i = 1; i <= workerPorts.length; i++) {
				long id = i;
				String address = "localhost:" + workerPorts[i - 1];
				workerMonitoringService.create(new WorkerModel(id, "Worker", address));
				collect.add(() -> StatisticsService.getWorkerStatistics(id, address));
			}

			long start = System.currentTimeMillis();

			// Returns a list of Futures holding their status and results when all complete.
			// Future.isDone() is true for each element of the returned list
			// https://docs.oracle.com/javase/7/docs/api/java/util/concurrent/ExecutorService.html#invokeAll(java.util.Collection)
			List<Future<StatisticsModel>> taskFutures = executor.invokeAll(collect);

			taskFutures.forEach(res -> parse.add(() -> syncWorkerStats(res.get(), res.get().traffic.get(0).workerId)));

			executor.invokeAll(parse);

			long finish = System.currentTimeMillis();
			long elapsedTime = (finish - start);

			if (j >= 0) {
				meanExecTime += elapsedTime;
			}
		}

		executor.shutdown();

		// Wait until all threads are finish
		// Returns true if all tasks have completed following shut down.
		// Note that isTerminated is never true unless either shutdown or shutdownNow was called first.
		while (!executor.isTerminated());

		LOG.info(String.format(PERFORMANCE_FORMAT, workerPorts.length, Math.round(meanExecTime / numRepetitionsExperiment)));
	}

	@Test
	public void testWorkerStatisticsReturnedForMonitoring() {
		workerMonitoringService.create(new WorkerModel(1L, "Worker", "localhost:" + workerPorts[0]));

		var model = workerMonitoringService.get(1L);

		Assert.assertNotNull("Stats field of model contains worker statistics", model);
	}

	@Test
	public void testNonExistentWorkerStatistics() {
		workerMonitoringService.create(new WorkerModel(1L, "Worker", "localhost:8001"));
		var options = new StatisticsOptions();
		options.utilization = true;

		var stats = statisticsMonitoringService.getAll(1L, options);

		Assert.assertEquals("Utilization field of model contains worker statistics", 0, stats.utilization.size());
	}

	private Boolean syncWorkerStats(StatisticsModel stats, Long id) {
		CompletableFuture<Boolean> utilizationFuture = null;
		CompletableFuture<Boolean> trafficFuture = null;
		CompletableFuture<Boolean> eventsFuture = null;
		CompletableFuture<Boolean> dataObjFuture = null;
		CompletableFuture<Boolean> requestsFuture = null;

		if (stats != null) {

			if (stats.utilization != null) {
				utilizationFuture = CompletableFuture.supplyAsync(() -> {
					entityRepository.createEntity(stats.utilization.get(0));
					return true;
				});
			}
			if (stats.traffic != null) {
				trafficFuture = CompletableFuture.supplyAsync(() -> {
					for (var trafficEntity : stats.traffic) {
						if (trafficEntity.coordinatorId > 0) {
							entityRepository.createEntity(trafficEntity);
						}
					}
					return true;
				});
			}
			if (stats.events != null) {
				eventsFuture = CompletableFuture.supplyAsync(() -> {
					for (var eventEntity: stats.events) {
						if (eventEntity.coordinatorId > 0) {
							var eventId = entityRepository.createEntity(eventEntity);

							for (var stageEntity : eventEntity.stages) {
								stageEntity.eventId = eventId;

								entityRepository.createEntity(stageEntity);
							}
						}
					}
					return true;
				});
			}
			if (stats.dataObjects != null) {
				dataObjFuture = CompletableFuture.supplyAsync(() -> {
					entityRepository.removeAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, id, DataObjectModel.class);

					for (var dataObjectEntity : stats.dataObjects) {
						entityRepository.createEntity(dataObjectEntity);
					}

					return true;
				});
			}
			if (stats.requests != null) {
				requestsFuture = CompletableFuture.supplyAsync(() -> {
					entityRepository.removeAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, id, RequestModel.class);

					for (var requestEntity : stats.requests) {
						if (requestEntity.coordinatorId > 0) {
							entityRepository.createEntity(requestEntity);
						}
					}

					return true;
				});
			}
		}
		List<CompletableFuture<Boolean>> completableFutures = Arrays.asList(utilizationFuture, trafficFuture, eventsFuture, dataObjFuture, requestsFuture);

		completableFutures.forEach(cf -> {
			try {
				cf.get();
			} catch (InterruptedException | ExecutionException e) {
				throw new RuntimeException(e);
			}
		});

		return true;
	}
}
