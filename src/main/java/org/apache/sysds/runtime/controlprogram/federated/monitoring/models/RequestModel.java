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

public class RequestModel extends CoordinatorConnectionModel {

	public Long workerId;
	public String type;
	public Long count;

	private static final String JsonFormat = "{" +
			"\"type\": \"%s\"," +
			"\"count\": %d" +
			"}";

	public RequestModel() {
		this(-1L);
	}

	private RequestModel(final Long id) {
		this.id = id;
	}

	public RequestModel(final String type, final Long count) {
		this(-1L, type, count);
	}

	public RequestModel(final Long id, final String type, final Long count) {
		this.id = id;
		this.type = type;
		this.count = count;
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, this.type, this.count);
	}
}
