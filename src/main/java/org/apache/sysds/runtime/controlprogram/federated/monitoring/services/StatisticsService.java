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

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.*;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.Constants;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Future;

public class StatisticsService {

	private static final IRepository entityRepository = new DerbyRepository();
	private static final Long endOfDynamicPorts = 65535L;
	private static final int maxMonitorHostCoordinators = DMLScript.MAX_MONITOR_HOST_COORDINATORS;

	public StatisticsModel getAll(Long workerId, StatisticsOptions options) {
		var stats = new StatisticsModel();

		if (options.utilization) {
			stats.utilization = entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, UtilizationModel.class, options.rowCount);
		}

		if (options.traffic) {
			stats.traffic = entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, TrafficModel.class, options.rowCount);
		}

		if (options.events) {
			stats.events = entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, EventModel.class, options.rowCount);

			for (var event: stats.events) {
				event.setCoordinatorName(entityRepository.getEntity(event.coordinatorId, CoordinatorModel.class).name);

				event.stages = entityRepository.getAllEntitiesByField(Constants.ENTITY_EVENT_ID_COL, event.id, EventStageModel.class);
			}
		}

		if (options.dataObjects) {
			stats.dataObjects = entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, DataObjectModel.class);
		}

		if (options.requests) {
			stats.requests = entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, RequestModel.class);
		}

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
		var utilization = new UtilizationModel(workerId, aggFedStats.cpuUsage, aggFedStats.memoryUsage);
		var traffic = aggFedStats.coordinatorsTrafficBytes;
		var events = aggFedStats.workerEvents;
		var dataObjects = aggFedStats.workerDataObjects;
		var requests = aggFedStats.workerRequests;

		traffic.forEach(t -> t.workerId = workerId);
		dataObjects.forEach(o -> o.workerId = workerId);
		requests.forEach(r -> r.workerId = workerId);

		for (var event: events) {
			event.workerId = workerId;

			setCoordinatorId(event);
		}

		for (var trafficEntry: traffic) {
			trafficEntry.workerId = workerId;

			setCoordinatorId(trafficEntry);
		}

		return new StatisticsModel(List.of(utilization), traffic, events, dataObjects, requests);
	}

	private static void setCoordinatorId(CoordinatorConnectionModel entity) {
		List<CoordinatorModel> coordinators = new ArrayList<>();
		var monitoringKey = getCoordinatorMonitoringKey(entity.getCoordinatorAddress());

		if (monitoringKey != null) {
			coordinators = entityRepository.getAllEntitiesByField(Constants.ENTITY_MONITORING_KEY_COL, monitoringKey, CoordinatorModel.class);
		}

		if (!coordinators.isEmpty()) {
			entity.coordinatorId = coordinators.get(0).id;
		} else {
			entity.coordinatorId = -1L;
		}
	}

	private static String getCoordinatorMonitoringKey(String address) {
		String result = null;
		if (address != null && !address.isEmpty() && !address.isBlank()) {
			var aggAddress = address.split(":");

			var host = aggAddress[0];
			var port = Integer.parseInt(aggAddress[1]);

			var model = new CoordinatorModel();
			model.host = host;
			model.monitoringId = (endOfDynamicPorts - port) % maxMonitorHostCoordinators;

			model.generateMonitoringKey();

			result = model.monitoringHostIdKey;
		}

		return result;
	}

	private static Future<FederatedResponse> sendStatisticsRequest(String address) {
		Future<FederatedResponse> result = null;
		String host = address.split(":")[0];
		int port = Integer.parseInt(address.split(":")[1]);

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

		return result;
	}
}
