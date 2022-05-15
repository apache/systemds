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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.netty.handler.codec.http.FullHttpResponse;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.Request;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.Response;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.BaseEntityModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.WorkerService;

import java.io.IOException;

public class WorkerController implements IController {

	private final WorkerService _workerService = new WorkerService();

	@Override
	public FullHttpResponse create(Request request) {

		ObjectMapper mapper = new ObjectMapper();

		try {
			BaseEntityModel model = mapper.readValue(request.getBody(), BaseEntityModel.class);
			_workerService.create(model);
			return Response.ok("Success");
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public FullHttpResponse update(Request request, Long objectId) {
		return null;
	}

	@Override
	public FullHttpResponse delete(Request request, Long objectId) {
		return null;
	}

	@Override
	public FullHttpResponse get(Request request, Long objectId) {
		var result = _workerService.get(objectId);

		if (result == null) {
			return Response.notFound("No such worker can be found");
		}

		return Response.ok(result.toString());
	}

	@Override
	public FullHttpResponse getAll(Request request) {
		var workers = _workerService.getAll();

		if (workers.isEmpty()) {
			return Response.notFound("No workers can be found");
		}

		return Response.ok(workers.toString());
	}
}
