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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.lang3.tuple.Triple;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

public class StatsEntityModel extends BaseEntityModel {
	private Long _workerId;
	private double _cpuUsage;
	private double _memoryUsage;
	private Map<String, Pair<Long, Double>> _heavyHitterInstructionsObj;
	private String _heavyHitterInstructions;
	private List<Triple<LocalDateTime, String, Long>> _transferredBytesObj;
	private String _transferredBytes;

	public StatsEntityModel() { }

	public StatsEntityModel(Long workerId, double cpuUsage, double memoryUsage,
		Map<String, Pair<Long, Double>> heavyHitterInstructionsObj,
		List<Triple<LocalDateTime, String, Long>> transferredBytesObj)
	{
		_workerId = workerId;
		_cpuUsage = cpuUsage;
		_memoryUsage = memoryUsage;
		_heavyHitterInstructionsObj = heavyHitterInstructionsObj;
		_transferredBytesObj = transferredBytesObj;
		_heavyHitterInstructions = "";
		_transferredBytes = "";
	}

	public Long getWorkerId() {
		return _workerId;
	}

	public void setWorkerId(final Long workerId) {
		_workerId = workerId;
	}

	public double getCPUUsage() {
		return _cpuUsage;
	}

	public void setCPUUsage(final double cpuUsage) {
		_cpuUsage = cpuUsage;
	}

	public double getMemoryUsage() {
		return _memoryUsage;
	}

	public void setMemoryUsage(final double memoryUsage) {
		_memoryUsage = memoryUsage;
	}

	public String getHeavyHitterInstructions() {
		if (_heavyHitterInstructions.isEmpty() || _heavyHitterInstructions.isBlank()) {
			StringBuilder sb = new StringBuilder();

			sb.append("{");
			for(Map.Entry<String, Pair<Long, Double>> entry : _heavyHitterInstructionsObj.entrySet()) {
				String instruction = entry.getKey();
				Long count = entry.getValue().getLeft();
				double duration = entry.getValue().getRight();
				sb.append(String.format("{" +
					"\"instruction\": %s," +
					"\"count\": \"%d\"," +
					"\"duration\": \"%.2f\"," +
					"},", instruction, count, duration));
			}
			sb.append("}");

			_heavyHitterInstructions = sb.toString();
		}

		return _heavyHitterInstructions;
	}

	public void setHeavyHitterInstructions(final String heavyHitterInstructionsJsonString) {
		_heavyHitterInstructions = heavyHitterInstructionsJsonString;
	}

	public String getTransferredBytes() {
		if (_transferredBytes.isEmpty() || _transferredBytes.isBlank()) {
			StringBuilder sb = new StringBuilder();

			sb.append("{");
			for (var entry: _transferredBytesObj) {
				sb.append(String.format("{" +
					"\"datetime\": %s," +
					"\"coordinatorAddress\": \"%s\"," +
					"\"byteAmount\": \"%d\"," +
					"},", entry.getLeft().format(DateTimeFormatter.ISO_DATE_TIME),
					entry.getMiddle(), entry.getRight()));
			}
			sb.append("}");

			_transferredBytes = sb.toString();
		}

		return _transferredBytes;
	}

	public void setTransferredBytes(final String transferredBytesJsonString) {
		_transferredBytes = transferredBytesJsonString;
	}

	@Override
	public String toString() {
		return String.format("{" +
			"\"cpuUsage\": %.2f," +
			"\"memoryUsage\": %.2f," +
			"\"coordinatorTraffic\": %s," +
			"\"heavyHitters\": %s" +
			"}", _cpuUsage, _memoryUsage, getTransferredBytes(), getHeavyHitterInstructions());
	}
}
