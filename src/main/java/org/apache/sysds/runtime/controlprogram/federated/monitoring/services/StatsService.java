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

import org.apache.log4j.Logger;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatsEntityModel;

import java.net.InetSocketAddress;
import java.sql.Timestamp;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class StatsService {
	protected static Logger log = Logger.getLogger(StatsService.class);

	public static BaseEntityModel getWorkerStatistics(Long id, String address) {
		StatsEntityModel parsedStats = null;

		try {
			var statisticsResponse = sendStatisticsRequest(address).get(5, TimeUnit.SECONDS);

			if (statisticsResponse.isSuccessful()) {
				FederatedStatistics.FedStatsCollection aggFedStats = new FederatedStatistics.FedStatsCollection();

				Object[] tmp = statisticsResponse.getData();
				if(tmp[0] instanceof FederatedStatistics.FedStatsCollection)
					aggFedStats.aggregate((FederatedStatistics.FedStatsCollection)tmp[0]);

				parsedStats = new StatsEntityModel(
						id,
						new Timestamp(System.currentTimeMillis()),
						aggFedStats.cpuUsage,
						aggFedStats.memoryUsage,
						aggFedStats.jitCompileTime,
						aggFedStats.heavyHitters,
						aggFedStats.coordinatorsTrafficBytes,
						aggFedStats.requestTypeCount);
			}
		} catch (DMLRuntimeException dre) {
			// silently ignore -> caused by offline federated workers
			log.error("Worker offline: " + dre.getMessage());
		} catch (Exception e) {
			log.error("Error: " + e.getMessage());
		}

		return parsedStats;
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
