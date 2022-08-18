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

public class TrafficModel extends CoordinatorConnectionModel {

	public Long workerId;
	public LocalDateTime timestamp;
	public Long byteAmount;

	private static final String JsonFormat = "{" +
			"\"timestamp\": \"%s\"," +
			"\"coordinatorId\": %d," +
			"\"byteAmount\": %d" +
			"}";

	public TrafficModel() { }

	public TrafficModel(final LocalDateTime timestamp, final String coordinatorHostId, final Long byteAmount) {
		this.timestamp = timestamp;
		this.byteAmount = byteAmount;

		super.setCoordinatorHostId(coordinatorHostId);
	}

	public TrafficModel(final Long workerId, final String coordinatorAddress, final Long byteAmount) {
		this(-1L, workerId, LocalDateTime.now(), coordinatorAddress, byteAmount);
	}

	public TrafficModel(final Long id, final Long workerId, final LocalDateTime timestamp, final String coordinatorHostId, final Long byteAmount) {
		this.id = id;
		this.workerId = workerId;
		this.timestamp = timestamp;
		this.byteAmount = byteAmount;

		super.setCoordinatorHostId(coordinatorHostId);
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, this.timestamp, this.coordinatorId, this.byteAmount);
	}
}
