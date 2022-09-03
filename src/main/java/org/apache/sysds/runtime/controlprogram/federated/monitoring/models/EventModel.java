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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.models;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class EventModel extends CoordinatorConnectionModel {
	private static final long serialVersionUID = -5597621916956632690L;
	public Long workerId;
	private String coordinatorName;
	public List<EventStageModel> stages;

	private static final String JsonFormat = "{" +
			"\"coordinatorName\": \"%s\"," +
			"\"stages\": [%s]" +
			"}";

	public EventModel() {
		this(-1L);
	}

	private EventModel(final Long id) {
		this.id = id;
		this.stages = new ArrayList<>();
	}

	public EventModel(final Long workerId, final Long coordinatorId) {
		this(-1L, workerId, coordinatorId);
	}

	public EventModel(final Long id, final Long workerId, final Long coordinatorId) {
		this.id = id;
		this.workerId = workerId;
		this.coordinatorId = coordinatorId;
		this.stages = new ArrayList<>();
	}

	public void setCoordinatorName(String name) {
		this.coordinatorName = name;
	}

	@Override
	public String toString() {
		String stagesStr = this.stages.stream()
				.map(EventStageModel::toString)
				.collect(Collectors.joining(","));

		return String.format(JsonFormat, this.coordinatorName, stagesStr);
	}
}
