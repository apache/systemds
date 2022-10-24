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

public class UtilizationModel extends BaseModel {
	private static final long serialVersionUID = 6984053518916899551L;
	public Long workerId;
	public LocalDateTime timestamp;
	public double cpuUsage;
	public double memoryUsage;

	private static final String JsonFormat = "{" +
			"\"timestamp\": \"%s\"," +
			"\"cpuUsage\": %.2f," +
			"\"memoryUsage\": %.2f" +
			"}";

	public UtilizationModel() {
		this(-1L);
	}

	private UtilizationModel(final Long id) {
		this.id = id;
	}

	public UtilizationModel(final double cpuUsage, final double memoryUsage) {
		this(-1L, -1L, LocalDateTime.now(), cpuUsage, memoryUsage);
	}

	public UtilizationModel(final Long workerId, final double cpuUsage, final double memoryUsage) {
		this(-1L, workerId, LocalDateTime.now(), cpuUsage, memoryUsage);
	}

	public UtilizationModel(final Long id, final Long workerId, final LocalDateTime timestamp, final double cpuUsage, final double memoryUsage) {
		this.id = id;
		this.workerId = workerId;
		this.timestamp = timestamp;
		this.cpuUsage = cpuUsage;
		this.memoryUsage = memoryUsage;
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, this.timestamp, this.cpuUsage, this.memoryUsage);
	}
}
