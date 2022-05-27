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

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.NodeEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.StatsEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.EntityEnum;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class WorkerService {
	private static final IRepository _entityRepository = new DerbyRepository();
	private final Map<Long, String> _cachedWorkers = new HashMap<>();

	public WorkerService() {
		Runnable syncWorkerStatisticsWithDBRunnable = this::syncWorkerStatisticsWithDB;

		ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
		executor.scheduleAtFixedRate(syncWorkerStatisticsWithDBRunnable, 0, 3, TimeUnit.SECONDS);
	}

	public void create(BaseEntityModel model) {
		long id = _entityRepository.createEntity(EntityEnum.WORKER, model);

		var modelEntity = (NodeEntityModel) model;

		_cachedWorkers.put(id, modelEntity.getAddress());
	}

	public BaseEntityModel get(Long id) {
		var model = (NodeEntityModel) _entityRepository.getEntity(EntityEnum.WORKER, id);
		var stats = (List<BaseEntityModel>) _entityRepository.getAllEntitiesByField(EntityEnum.WORKER_STATS, id);

		model.setStats(stats);

		return model;
	}

	public List<BaseEntityModel> getAll() {
		var workersRaw = _entityRepository.getAllEntities(EntityEnum.WORKER);
		var workersResult = new ArrayList<BaseEntityModel>();

		for (var worker: workersRaw) {
			var workerModel = (NodeEntityModel) worker;
			var stats = (List<BaseEntityModel>) _entityRepository.getAllEntitiesByField(EntityEnum.WORKER_STATS, workerModel.getId());

			workerModel.setStats(stats);

			workersResult.add(workerModel);
		}

		return workersResult;
	}

	private void syncWorkerStatisticsWithDB() {
		for(Map.Entry<Long, String> entry : _cachedWorkers.entrySet()) {
			Long id = entry.getKey();
			String address = entry.getValue();

			var stats = (StatsEntityModel) StatsService.getWorkerStatistics(id, address);

			_entityRepository.createEntity(EntityEnum.WORKER_STATS, stats);
		}
	}
}
