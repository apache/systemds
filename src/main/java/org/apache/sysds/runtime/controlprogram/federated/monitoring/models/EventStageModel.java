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

import java.time.LocalDateTime;

public class EventStageModel extends BaseModel {

	public Long eventId;
	public String operation;
	public LocalDateTime startTime;
	public LocalDateTime endTime;

	private static final String JsonFormat = "{" +
			"\"operation\": \"%s\"," +
			"\"startTime\": \"%s\"," +
			"\"endTime\": \"%s\"" +
			"}";

	public EventStageModel() {
		this(-1L);
	}

	private EventStageModel(final Long id) {
		this.id = id;
	}

	public EventStageModel(final Long eventId, final String stageOperation) {
		this(-1L, eventId, stageOperation);
	}

	public EventStageModel(final Long id, final Long eventId, final String operation) {
		this.id = id;
		this.eventId = eventId;
		this.operation = operation;
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, this.operation, this.startTime, this.endTime);
	}
}
