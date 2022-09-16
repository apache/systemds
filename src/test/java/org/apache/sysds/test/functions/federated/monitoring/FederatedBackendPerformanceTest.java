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

import static java.lang.Thread.sleep;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.net.http.HttpResponse;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class FederatedBackendPerformanceTest extends FederatedMonitoringTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedBackendPerformanceTest.class.getName());
	private final static String TEST_NAME = "FederatedBackendPerformanceTest";
	private final static String TEST_DIR = "functions/federated/monitoring/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedBackendPerformanceTest.class.getSimpleName() + "/";
	private static final String PERFORMANCE_FORMAT = "For %d number of requests, milliseconds elapsed %d.";

	private static int[] workerPort;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
		startFedMonitoring(null);
		workerPort = startFedWorkers(1);
	}

	@Test
	@Ignore
	public void testBackendPerformance() throws InterruptedException {
		int numRequests = 20;

		double meanExecTime = 0.f;
		double numRepetitionsExperiment = 100.f;

		addEntities(1, Entity.WORKER);
		updateEntity(new WorkerModel(1L, "Worker", "localhost:" + workerPort[0]), Entity.WORKER);
		// Give time for statistics to be collected (70s)
		sleep(70000);

		ExecutorService executor = Executors.newFixedThreadPool(numRequests);

		for (int j = -10; j < numRepetitionsExperiment; j++) {

			long start = System.currentTimeMillis();

			// Returns a list of Futures holding their status and results when all complete.
			// Future.isDone() is true for each element of the returned list
			// https://docs.oracle.com/javase/7/docs/api/java/util/concurrent/ExecutorService.html#invokeAll(java.util.Collection)
			List<Future<HttpResponse<?>>> taskFutures = executor.invokeAll(Collections.nCopies(numRequests,
					() -> getEntities(Entity.STATISTICS)));

			long finish = System.currentTimeMillis();
			long elapsedTime = (finish - start);

			if (j >= 0) {
				meanExecTime += elapsedTime;
			}

			taskFutures.forEach(res -> {
				try {
					Assert.assertEquals("Stats parsed correctly", res.get().statusCode(), 200);
				} catch (InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
			});

			// Wait for a second at the end of each iteration
			sleep(500);
		}

		executor.shutdown();

		// Wait until all threads are finished
		// Returns true if all tasks have completed following shut down.
		// Note that isTerminated is never true unless either shutdown or shutdownNow was called first.
		while (!executor.isTerminated());

		LOG.info(String.format(PERFORMANCE_FORMAT, numRequests, Math.round(meanExecTime / numRepetitionsExperiment)));
	}
}
