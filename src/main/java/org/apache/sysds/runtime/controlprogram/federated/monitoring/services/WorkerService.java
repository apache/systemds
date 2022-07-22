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

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatisticsModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class WorkerService {
	private static final IRepository _entityRepository = new DerbyRepository();
	private static final Map<Long, String> _cachedWorkers = new HashMap<>();

	public WorkerService() {
		updateCachedWorkers(null);

		ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
		executor.scheduleAtFixedRate(syncWorkerStatisticsWithDB(), 0, 3, TimeUnit.SECONDS);
	}

	public void create(WorkerModel model) {
		long id = _entityRepository.createEntity(model);

		_cachedWorkers.putIfAbsent(id, model.address);
	}

	public void update(BaseModel model) {
		_entityRepository.updateEntity(model);
	}

	public void remove(Long id) {
		_entityRepository.removeEntity(id, WorkerModel.class);

		_cachedWorkers.remove(id);
	}

	public WorkerModel get(Long id) {
		var model = _entityRepository.getEntity(id, WorkerModel.class);

		updateCachedWorkers(null);

		return model;
	}

	public List<WorkerModel> getAll() {
		var workers = _entityRepository.getAllEntities(WorkerModel.class);

		updateCachedWorkers(workers);

		return workers;
	}

	private void updateCachedWorkers(List<WorkerModel> workersRaw) {
		List<WorkerModel> workersTmp = workersRaw;

		if (workersTmp == null) {
			workersTmp = getAll();
		}

		for(var worker : workersTmp) {
			_cachedWorkers.putIfAbsent(worker.id, worker.address);
		}
	}

	private static Runnable syncWorkerStatisticsWithDB() {
		return () -> {

			for(Map.Entry<Long, String> entry : _cachedWorkers.entrySet()) {
				Long id = entry.getKey();
				String address = entry.getValue();

				var stats = StatisticsService.getWorkerStatistics(id, address);

				if (stats.utilization != null) {
					_entityRepository.createEntity(stats.utilization.get(0));
				}
				if (stats.traffic != null) {
					for (var trafficEntity: stats.traffic) {
						_entityRepository.createEntity(trafficEntity);
					}
				}
				if (stats.jobs != null) {
					for (var jobEntity: stats.jobs) {
						var jobId = _entityRepository.createEntity(jobEntity);

						for (var stageEntity: jobEntity.stages) {
							stageEntity.jobId = jobId;
							System.out.println(stageEntity);

							_entityRepository.createEntity(stageEntity);
						}
					}
				}
			}
		};
	}
}
