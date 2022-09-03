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

public abstract class CoordinatorConnectionModel extends BaseModel {
	private static final long serialVersionUID = 918360814223266197L;
	public Long coordinatorId;
	private String coordinatorHostId;
	private static final String localhostIp = "127.0.0.1";
	private static final String localhostString = "localhost";

	public CoordinatorConnectionModel() { }

	public void setCoordinatorHostId(String hostId) {
		this.coordinatorHostId = hostId;
	}

	public String getCoordinatorHostId() {
		this.coordinatorHostId = this.coordinatorHostId.replaceFirst("/", "");

		if (this.coordinatorHostId.contains(localhostIp)) {
			this.coordinatorHostId = this.coordinatorHostId.replace(localhostIp, localhostString);
		}

		this.coordinatorHostId = this.coordinatorHostId.replaceFirst(":\\d+", "");

		return this.coordinatorHostId;
	}
}
