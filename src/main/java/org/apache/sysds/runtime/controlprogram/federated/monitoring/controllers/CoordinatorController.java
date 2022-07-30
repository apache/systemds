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

import io.netty.handler.codec.http.FullHttpResponse;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.models.CoordinatorModel;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Request;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.Response;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.CoordinatorService;
import org.apache.sysds.runtime.controlprogram.federated.monitoring.services.MapperService;

public class CoordinatorController implements IController {
	private final CoordinatorService coordinatorService = new CoordinatorService();

	@Override
	public FullHttpResponse create(Request request) {

		var model = MapperService.getModelFromBody(request, CoordinatorModel.class);
		model.generateMonitoringKey();

		model.id = coordinatorService.create(model);

		return Response.ok(model.toString());
	}

	@Override
	public FullHttpResponse update(Request request, Long objectId) {
		var model = MapperService.getModelFromBody(request, CoordinatorModel.class);
		model.generateMonitoringKey();

		coordinatorService.update(model);

		return Response.ok(model.toString());
	}

	@Override
	public FullHttpResponse delete(Request request, Long objectId) {
		coordinatorService.remove(objectId);

		return Response.ok(Constants.GENERIC_SUCCESS_MSG);
	}

	@Override
	public FullHttpResponse get(Request request, Long objectId) {
		var result = coordinatorService.get(objectId);

		if (result == null) {
			return Response.notFound(Constants.NOT_FOUND_MSG);
		}

		return Response.ok(result.toString());
	}

	@Override
	public FullHttpResponse getAll(Request request) {
		var coordinators = coordinatorService.getAll();

		return Response.ok(coordinators.toString());
	}
}
