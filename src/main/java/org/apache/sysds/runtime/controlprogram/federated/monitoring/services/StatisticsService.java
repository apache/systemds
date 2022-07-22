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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.JobModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.JobStageModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsOptions;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.TrafficModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.UtilizationModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.Constants;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

import java.net.InetSocketAddress;
import java.util.List;
import java.util.concurrent.Future;

public class StatisticsService {

	private static final IRepository _entityRepository = new DerbyRepository();

	public StatisticsModel getAll(Long workerId, StatisticsOptions options) {
		var stats = new StatisticsModel();

		if (options.utilization) {
			stats.utilization = _entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, UtilizationModel.class, options.rowCount);
		}

		if (options.traffic) {
			stats.traffic = _entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, TrafficModel.class, options.rowCount);
		}

		if (options.jobs) {
			stats.jobs = _entityRepository.getAllEntitiesByField(Constants.ENTITY_WORKER_ID_COL, workerId, JobModel.class, options.rowCount);

			for (var job: stats.jobs) {
				job.stages = _entityRepository.getAllEntitiesByField(Constants.ENTITY_JOB_ID_COL, job.id, JobStageModel.class);
			}
		}

		return stats;
	}

	public static StatisticsModel getWorkerStatistics(Long id, String address) {
		StatisticsModel parsedStats = null;

		try {
			var statisticsResponse = sendStatisticsRequest(address).get();

			if (statisticsResponse.isSuccessful()) {
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
		var jobs = aggFedStats.workerJobs;

		for (var entry: traffic) {
			entry.workerId = workerId;
		}

		for (var job: jobs) {
			job.workerId = workerId;

			var coordinators =
					_entityRepository.getAllEntitiesByField(Constants.ENTITY_ADDRESS_COL, job.getCoordinatorAddress(), CoordinatorModel.class);

			if (!coordinators.isEmpty()) {
				job.coordinatorId = coordinators.get(0).id;
			} else {
				job.coordinatorId = -1L;
			}
		}

		return new StatisticsModel(List.of(utilization), traffic, jobs);
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
