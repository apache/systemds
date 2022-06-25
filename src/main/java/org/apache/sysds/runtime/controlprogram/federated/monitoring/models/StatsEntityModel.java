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
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.sql.Timestamp;

public class StatsEntityModel extends BaseEntityModel {
	private Timestamp _timeStamp;
	private Long _workerId;
	private double _cpuUsage;
	private double _memoryUsage;
	private double _jitCompileTime = 0;
	private Map<String, Pair<Long, Double>> _heavyHitterInstructionsObj;
	private String _heavyHitterInstructions;
	private List<Triple<LocalDateTime, String, Long>> _transferredBytesObj;
	private String _transferredBytes;
	private List<Pair<FederatedRequest.RequestType, Long>> _requestTypeCount;

	public StatsEntityModel() { }

	public StatsEntityModel(
			Long workerId,
			Timestamp timestamp,
			double cpuUsage,
			double memoryUsage,
			double jitCompileTime,
			Map<String, Pair<Long, Double>> heavyHitterInstructionsObj,
			List<Triple<LocalDateTime, String, Long>> transferredBytesObj,
			List<Pair<FederatedRequest.RequestType, Long>> requestTypeCount)
	{
		_workerId = workerId;
		_timeStamp = timestamp;
		_cpuUsage = cpuUsage;
		_memoryUsage = memoryUsage;
		_jitCompileTime = jitCompileTime;
		_heavyHitterInstructionsObj = heavyHitterInstructionsObj;
		_transferredBytesObj = transferredBytesObj;
		_heavyHitterInstructions = "";
		_transferredBytes = "";
		_requestTypeCount = requestTypeCount;
	}

	public Long getWorkerId() {
		return _workerId;
	}

	public void setWorkerId(final Long workerId) {
		_workerId = workerId;
	}

	public Timestamp getTimestamp() {
		return _timeStamp;
	}

	public void setTimestamp(Timestamp timestamp) {
		_timeStamp = timestamp;
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
	public double getJitCompileTime() {
		return _jitCompileTime;
	}

	public List<Pair<FederatedRequest.RequestType, Long>> getRequestTypeCount() {
		return _requestTypeCount;
	}

	public String getHeavyHitterInstructions() {
		if ((_heavyHitterInstructions.isEmpty() || _heavyHitterInstructions.isBlank()) && _heavyHitterInstructionsObj != null) {

			List<String> heavyHittersStrArr = new ArrayList<>();

			for(Map.Entry<String, Pair<Long, Double>> entry : _heavyHitterInstructionsObj.entrySet()) {
				String instruction = entry.getKey();
				Long count = entry.getValue().getLeft();
				double duration = entry.getValue().getRight();
				heavyHittersStrArr.add(String.format(Constants.HEAVY_HITTER_INSTRUCTIONS_JSON_STR, instruction, count, duration));
			}

			if (!heavyHittersStrArr.isEmpty()) {
				_heavyHitterInstructions = String.join(",", heavyHittersStrArr);
			}
		}

		return _heavyHitterInstructions;
	}

	public void setHeavyHitterInstructions(final String heavyHitterInstructionsJsonString) {
		_heavyHitterInstructions = heavyHitterInstructionsJsonString;
	}

	public String getTransferredBytes() {
		if ((_transferredBytes.isEmpty() || _transferredBytes.isBlank()) && _transferredBytesObj != null) {

			List<String> transferredBytesStrArr = new ArrayList<>();

			for (var entry: _transferredBytesObj) {
				transferredBytesStrArr.add(String.format(Constants.TRANSFERRED_BYTES_JSON_STR, entry.getLeft().format(DateTimeFormatter.ISO_DATE_TIME),
					entry.getMiddle(), entry.getRight()));
			}

			if (!transferredBytesStrArr.isEmpty()) {
				_transferredBytes = String.join(",", transferredBytesStrArr);
			}
		}

		return _transferredBytes;
	}

	public void setTransferredBytes(final String transferredBytesJsonString) {
		_transferredBytes = transferredBytesJsonString;
	}

	@Override
	public String toString() {
		return String.format(Constants.STATS_ENTITY_JSON_STR, _timeStamp.toString(), _cpuUsage, _memoryUsage,
				getTransferredBytes(), getHeavyHitterInstructions());
	}
}
