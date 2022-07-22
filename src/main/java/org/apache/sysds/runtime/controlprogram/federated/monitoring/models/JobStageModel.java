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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class JobStageModel extends BaseModel implements Serializable {

	public Long jobId;
	public String stageType;
	public LocalDateTime startTime;
	public LocalDateTime endTime;
	public String data;
	private final List<Pair<String, LocalDateTime>> instructions;

	public JobStageModel() {
		this(-1L);
	}

	private JobStageModel(final Long id) {
		this.id = id;
		this.instructions = new ArrayList<>();
	}

	public JobStageModel(final Long jobId, final String stageType, final String data) {
		this(-1L, jobId, stageType, data);
	}

	public JobStageModel(final Long id, final Long jobId, final String stageType, final String data) {
		this.id = id;
		this.jobId = jobId;
		this.stageType = stageType;
		this.data = data;
		this.instructions = new ArrayList<>();
	}

	public void addInstructionToStage(String instruction, LocalDateTime startTime) {
		this.instructions.add(new ImmutablePair<>(instruction, startTime));

		this.data = this.instructions.stream()
				.map(i -> String.format("{" +
					"\"instruction\": \"%s\"," +
					"\"startTime\": %s" +
					"}", i.getLeft(), i.getRight()))
				.collect(Collectors.joining(","));
	}

	@Override
	public String toString() {
		return String.format("{" +
				"\"stageType\": \"%s\"," +
				"\"startTime\": %s," +
				"\"endTime\": %s," +
				"\"data\": [%s]" +
				"}", this.stageType, this.startTime, this.endTime, this.data);
	}
}
