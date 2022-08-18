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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.controllers;

import org.apache.sysds.runtime.controlprogram.federated.monitoring.Request;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Response;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.WorkerModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.MapperService;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.WorkerService;

import io.netty.handler.codec.http.FullHttpResponse;

public class WorkerController implements IController {
	private final WorkerService workerService = new WorkerService();

	@Override
	public FullHttpResponse create(Request request) {

		var model = MapperService.getModelFromBody(request, WorkerModel.class);

		model.id = workerService.create(model);

		return Response.ok(model.toString());
	}

	@Override
	public FullHttpResponse update(Request request, Long objectId) {
		var model = MapperService.getModelFromBody(request, WorkerModel.class);

		workerService.update(model);
		model.setOnlineStatus(workerService.getWorkerOnlineStatus(model.id));

		return Response.ok(model.toString());
	}

	@Override
	public FullHttpResponse delete(Request request, Long objectId) {
		workerService.remove(objectId);

		return Response.ok(Constants.GENERIC_SUCCESS_MSG);
	}

	@Override
	public FullHttpResponse get(Request request, Long objectId) {
		var result = workerService.get(objectId);

		if (result == null) {
			return Response.notFound(Constants.NOT_FOUND_MSG);
		}

		result.setOnlineStatus(workerService.getWorkerOnlineStatus(result.id));

		return Response.ok(result.toString());
	}

	@Override
	public FullHttpResponse getAll(Request request) {
		var workers = workerService.getAll();

		for (var worker: workers) {
			worker.setOnlineStatus(workerService.getWorkerOnlineStatus(worker.id));
		}

		return Response.ok(workers.toString());
	}
}
