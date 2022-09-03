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

public class CoordinatorModel extends BaseModel {
	private static final long serialVersionUID = 4116787631938152573L;
	public String name;
	public String host;
	public Long processId;
	public String monitoringHostIdKey;

	private static final String keyFormat = "%s-%d";

	private static final String JsonFormat = "{" +
			"\"id\": %d," +
			"\"name\": \"%s\"," +
			"\"host\": \"%s\"," +
			"\"processId\": %d" +
			"}";

	public CoordinatorModel(final Long id) {
		this.id = id;
	}

	public CoordinatorModel() {
		this(-1L);
	}

	public void generateMonitoringKey() {
		this.monitoringHostIdKey = String.format(keyFormat, host, processId);
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, super.id, this.name, this.host, this.processId);
	}
}
