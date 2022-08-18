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

public class DataObjectModel extends BaseModel {

	public Long workerId;
	public String varName;
	public String dataType;
	public String valueType;
	public Long size;

	private static final String JsonFormat = "{" +
			"\"varName\": \"%s\"," +
			"\"dataType\": \"%s\"," +
			"\"valueType\": \"%s\"," +
			"\"size\": %d" +
			"}";

	public DataObjectModel() {
		this(-1L);
	}

	private DataObjectModel(final Long id) {
		this.id = id;
	}

	public DataObjectModel(final String varName, final String dataType, final String valueType, final Long size) {
		this(-1L, varName, dataType, valueType, size);
	}

	public DataObjectModel(final Long id, final String varName, final String dataType, final String valueType, final Long size) {
		this.id = id;
		this.varName = varName;
		this.dataType = dataType;
		this.valueType = valueType;
		this.size = size;
	}

	@Override
	public String toString() {
		return String.format(JsonFormat, this.varName, this.dataType, this.valueType, this.size);
	}
}
