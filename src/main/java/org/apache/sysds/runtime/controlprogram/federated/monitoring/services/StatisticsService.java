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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.services;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorConnectionModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.DataObjectModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.EventStageModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.RequestModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsOptions;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.TrafficModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.UtilizationModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.Constants;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

public class StatisticsService {

	private static final IRepository entityRepository = new DerbyRepository();

	public StatisticsModel getAll(Long workerId, StatisticsOptions options) {
		CompletableFuture<Void> utilizationFuture = null;
		CompletableFuture<Void> trafficFuture = null;
		CompletableFuture<Void> eventsFuture = null;
		CompletableFuture<Void> dataObjFuture = null;
		CompletableFuture<Void> requestsFuture = null;

		var stats = new StatisticsModel();

		if (options.utilization) {
			utilizationFuture = CompletableFuture
					.supplyAsync(() -> entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, UtilizationModel.class, options.rowCount))
					.thenAcceptAsync(result -> stats.utilization = result);
		}

		if (options.traffic) {
			trafficFuture = CompletableFuture
					.supplyAsync(() -> entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, TrafficModel.class, options.rowCount))
					.thenAcceptAsync(result -> stats.traffic = result);
		}

		if (options.events) {
			eventsFuture = CompletableFuture
					.supplyAsync(() -> {
						var events = entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, EventModel.class, options.rowCount);

						for (var event : events) {
							event.setCoordinatorName(entityRepository.getEntity(event.coordinatorId, CoordinatorModel.class).name);

							event.stages = entityRepository.getAllEntitiesByField(Constants.ENTITY_EVENT_ID_COL, event.id, EventStageModel.class);
						}

						return events;
					})
					.thenAcceptAsync(result -> stats.events = result);
		}

		if (options.dataObjects) {
			dataObjFuture = CompletableFuture
					.supplyAsync(() -> entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, DataObjectModel.class))
					.thenAcceptAsync(result -> stats.dataObjects = result);
		}

		if (options.requests) {
			requestsFuture = CompletableFuture
					.supplyAsync(() -> entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, RequestModel.class))
					.thenAcceptAsync(result -> stats.requests = result);
		}

		List<CompletableFuture<Void>> completableFutures = Arrays.asList(utilizationFuture, trafficFuture, eventsFuture, dataObjFuture, requestsFuture);

		completableFutures.forEach(cf -> {
			try {
				cf.get();
			} catch (InterruptedException | ExecutionException e) {
				throw new RuntimeException(e);
			}
		});

		return stats;
	}

	public static StatisticsModel getWorkerStatistics(Long id, String address) {
		StatisticsModel parsedStats = null;

		try {
			FederatedResponse statisticsResponse = null;

			var statisticsResponseFuture = sendStatisticsRequest(address);

			if (statisticsResponseFuture != null) {
				statisticsResponse = statisticsResponseFuture.get();
			}

			if (statisticsResponse != null && statisticsResponse.isSuccessful()) {
				FederatedStatistics.FedStatsCollection aggFedStats = new FederatedStatistics.FedStatsCollection();

				Object[] tmp = statisticsResponse.getData();
				if(tmp[0] instanceof FederatedStatistics.FedStatsCollection)
					aggFedStats.aggregate((FederatedStatistics.FedStatsCollection)tmp[0]);

				parsedStats = parseStatistics(id, aggFedStats);
			}
		} catch(DMLRuntimeException dre) {
			// silently ignore -> caused by offline federated workers
		} catch (Exception e) {
			throw new RuntimeException(e);
		}

		return parsedStats;
	}

	private static StatisticsModel parseStatistics(Long workerId, FederatedStatistics.FedStatsCollection aggFedStats) {
		var utilization = aggFedStats.utilization;
		var traffic = aggFedStats.coordinatorsTrafficBytes;
		var events = aggFedStats.workerEvents;
		var dataObjects = aggFedStats.workerDataObjects;
		var requests = aggFedStats.workerRequests;

		utilization.workerId = workerId;
		traffic.forEach(t -> t.workerId = workerId);
		dataObjects.forEach(o -> o.workerId = workerId);

		for (var event: events) {
			event.workerId = workerId;

			setCoordinatorId(event);
		}

		for (var trafficEntry: traffic) {
			trafficEntry.workerId = workerId;

			setCoordinatorId(trafficEntry);
		}

		for (var request: requests) {
			request.workerId = workerId;

			setCoordinatorId(request);
		}

		return new StatisticsModel(List.of(utilization), traffic, events, dataObjects, requests);
	}

	private static void setCoordinatorId(CoordinatorConnectionModel entity) {
		List<CoordinatorModel> coordinators = new ArrayList<>();
		var monitoringKey = entity.getCoordinatorHostId();

		if (monitoringKey != null) {
			coordinators = entityRepository.getAllEntitiesByField(Constants.ENTITY_MONITORING_KEY_COL, monitoringKey, CoordinatorModel.class);
		}

		if (!coordinators.isEmpty()) {
			entity.coordinatorId = coordinators.get(0).id;
		} else {
			entity.coordinatorId = -1L;
		}
	}

	private static Future<FederatedResponse> sendStatisticsRequest(String address) {
		Future<FederatedResponse> result = null;

		final Pattern pattern = Pattern.compile("(.*://)?([A-Za-z0-9\\-\\.]+)(:[0-9]+)?(.*)");
		final Matcher matcher = pattern.matcher(address);

		if (matcher.find()) {
			String host = matcher.group(2);
			String portStr = matcher.group(3);
			int port = 80;

			if (portStr != null && !portStr.isBlank() && !portStr.isEmpty())
				port = Integer.parseInt(portStr.replace(":", ""));

			InetSocketAddress isa = new InetSocketAddress(host, port);
			FederatedRequest frUDF = new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF, -1,
					new FederatedStatistics.FedStatsCollectFunction());

			try {
				result = FederatedData.executeFederatedOperation(isa, frUDF);
			} catch(DMLRuntimeException dre) {
				throw dre; // caused by offline federated workers
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}

		return result;
	}
}
