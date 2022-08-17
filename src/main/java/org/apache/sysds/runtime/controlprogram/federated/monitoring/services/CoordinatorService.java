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

import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.DerbyRepository;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories.IRepository;

import java.util.List;

public class CoordinatorService {
	private static final IRepository entityRepository = new DerbyRepository();

	public Long create(CoordinatorModel model) {
		return entityRepository.createEntity(model);
	}

	public void update(CoordinatorModel model) {
		entityRepository.updateEntity(model);
	}

	public void remove(Long id) {
		entityRepository.removeEntity(id, CoordinatorModel.class);
	}

	public CoordinatorModel get(Long id) {
		return entityRepository.getEntity(id, CoordinatorModel.class);
	}

	public List<CoordinatorModel> getAll() {
		return entityRepository.getAllEntities(CoordinatorModel.class);
	}
}
