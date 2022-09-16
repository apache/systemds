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

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventStageModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsOptions;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.StatisticsService;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.WorkerService;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static java.lang.Thread.sleep;

public class FederatedWorkerStatisticsTest extends FederatedMonitoringTestBase {
	private final static String TEST_NAME = "FederatedWorkerStatisticsTest";

	private final static String TEST_DIR = "functions/federated/monitoring/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedWorkerStatisticsTest.class.getSimpleName() + "/";

	private static final String PERFORMANCE_FORMAT = "For %d number of requests, milliseconds elapsed %d.";

	private static int[] workerPorts;
	private final WorkerService workerMonitoringService = new WorkerService();
	private final StatisticsService statisticsMonitoringService = new StatisticsService();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
		workerPorts = startFedWorkers(3);
	}

	@Test
	public void testWorkerStatisticsParsedCorrectly() {

		var model = (StatisticsModel) StatisticsService.getWorkerStatistics(1L, "localhost:" + workerPorts[0]);

		Assert.assertNotNull("Stats parsed correctly", model);
		Assert.assertNotEquals("Utilization stats parsed correctly", 0, model.utilization.size());
	}

	@Test
	public void testWorkerStatisticsPerformance() throws InterruptedException {
		int numRequests = 60;

		ExecutorService executor = Executors.newFixedThreadPool(numRequests);

		long start = System.currentTimeMillis();

		// Returns a list of Futures holding their status and results when all complete.
		// Future.isDone() is true for each element of the returned list
		// https://docs.oracle.com/javase/7/docs/api/java/util/concurrent/ExecutorService.html#invokeAll(java.util.Collection)
		List<Future<StatisticsModel>> taskFutures = executor.invokeAll(Collections.nCopies(numRequests,
				() -> StatisticsService.getWorkerStatistics(1L, "localhost:" + workerPorts[0])));

		long finish = System.currentTimeMillis();
		long elapsedTime = (finish - start);

		taskFutures.forEach(res -> {
			try {
				Assert.assertNotNull("Stats parsed correctly", res.get());
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
		});

		executor.shutdown();

		// Wait until all threads are finish
		// Returns true if all tasks have completed following shut down.
		// Note that isTerminated is never true unless either shutdown or shutdownNow was called first.
		while (!executor.isTerminated());

		System.out.println(String.format(PERFORMANCE_FORMAT, numRequests, elapsedTime));
	}

	@Test
	public void testWorkerStatisticsReturnedForMonitoring() {
		workerMonitoringService.create(new WorkerModel(1L, "Worker", "localhost:" + workerPorts[0]));

		var model = workerMonitoringService.get(1L);

		Assert.assertNotNull("Stats field of model contains worker statistics", model);
	}

	@Test
	public void testNonExistentWorkerStatistics() {
		var bla = new EventModel(1L, -1L);
		var derby = new DerbyRepository();

		derby.createEntity(bla);
		derby.createEntity(bla);
		var in3 = derby.createEntity(bla);
		derby.createEntity(bla);

		derby.getEntity(in3, EventModel.class);

		new EventStageModel();


		workerMonitoringService.create(new WorkerModel(1L, "Worker", "localhost:8001"));
		var options = new StatisticsOptions();
		options.utilization = true;

		var stats = statisticsMonitoringService.getAll(1L, options);

		Assert.assertEquals("Utilization field of model contains worker statistics", 0, stats.utilization.size());
	}
}
