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

public class HeavyHitterModel extends BaseModel {
	private static final long serialVersionUID = 1L;
	public Long workerId;
	public String operation;
	public double duration;
	public Long count;

	private static final String JsonFormat = "{" +
			"\"operation\": \"%s\"," +
			"\"duration\": %.2f," +
			"\"count\": %d" +
			"}";

	public HeavyHitterModel() {
		this(-1L);
	}

	private HeavyHitterModel(final Long id) {
		this.id = id;
	}

	public HeavyHitterModel(final Long workerId,
							final String operation,
							final double duration,
							final Long count) {
		this(-1L, workerId, operation, duration, count);
	}

	public HeavyHitterModel(final Long id,
							final Long workerId,
							final String operation,
							final double duration,
							final Long count) {
		this.id = id;
		this.workerId = workerId;
		this.operation = operation;
		this.duration = duration;
		this.count = count;
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, this.operation, this.duration, this.count);
	}
}
