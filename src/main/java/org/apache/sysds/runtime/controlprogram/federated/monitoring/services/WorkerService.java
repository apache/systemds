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
	private static final Map<Long, String> _cachedWorkers = new HashMap<>();

	public WorkerService() {
		updateCachedWorkers(null);

		ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
		executor.scheduleAtFixedRate(syncWorkerStatisticsWithDB(), 0, 3, TimeUnit.SECONDS);
	}

	public Long create(BaseEntityModel model) {
		long id = _entityRepository.createEntity(EntityEnum.WORKER, model);

		var modelEntity = (NodeEntityModel) model;

		_cachedWorkers.putIfAbsent(id, modelEntity.getAddress());

		return id;
	}

	public void update(BaseEntityModel model) {
		_entityRepository.updateEntity(EntityEnum.WORKER, model);

		NodeEntityModel editModel = (NodeEntityModel) model;

		_cachedWorkers.replace(editModel.getId(), editModel.getAddress());
	}

	public void remove(Long id) {
		_entityRepository.removeEntity(EntityEnum.WORKER, id);

		_cachedWorkers.remove(id);
	}

	public BaseEntityModel get(Long id) {
		var model = (NodeEntityModel) _entityRepository.getEntity(EntityEnum.WORKER, id);

		updateCachedWorkers(null);

		updateWorkersStats(model);

		return model;
	}

	public List<BaseEntityModel> getAll() {
		var workersRaw = _entityRepository.getAllEntities(EntityEnum.WORKER);
		var workersResult = new ArrayList<BaseEntityModel>();

		updateCachedWorkers(workersRaw);

		for (var worker: workersRaw) {
			var workerModel = (NodeEntityModel) worker;

			updateWorkersStats(workerModel);

			workersResult.add(workerModel);
		}

		return workersResult;
	}

	private void updateWorkersStats(NodeEntityModel model) {
		var savedStats = (List<BaseEntityModel>) _entityRepository.getAllEntitiesByField(EntityEnum.WORKER_STATS, model.getId());
		var recentStats = (StatsEntityModel) StatsService.getWorkerStatistics(model.getId(), model.getAddress());

		model.setOnlineStatus(recentStats != null);

		if (recentStats != null) {
			model.setJitCompileTime(recentStats.getJitCompileTime());
			model.setRequestTypeCount(recentStats.getRequestTypeCount());
		}

		model.setStats(savedStats);
	}

	private void updateCachedWorkers(List<BaseEntityModel> workersRaw) {
		List<BaseEntityModel> workersBaseModel = workersRaw;

		if (workersBaseModel == null) {
			workersBaseModel = getAll();
		}

		for(var workerBaseModel : workersBaseModel) {
			var worker = (NodeEntityModel) workerBaseModel;

			_cachedWorkers.putIfAbsent(worker.getId(), worker.getAddress());
		}
	}

	private static Runnable syncWorkerStatisticsWithDB() {
		return () -> {
			for(Map.Entry<Long, String> entry : _cachedWorkers.entrySet()) {
				Long id = entry.getKey();
				String address = entry.getValue();

				var stats = (StatsEntityModel) StatsService.getWorkerStatistics(id, address);

				if (stats != null) {
					_entityRepository.createEntity(EntityEnum.WORKER_STATS, stats);
				}
			}
		};
	}
}
