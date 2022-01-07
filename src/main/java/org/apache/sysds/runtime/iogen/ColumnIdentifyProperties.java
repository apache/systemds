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

package org.apache.sysds.runtime.iogen;

public class ColumnIdentifyProperties {

	private String indexPositionDelimiter;
	private Integer indexPosition;
	private String valueEndWithString;

	public ColumnIdentifyProperties() {
	}

	public ColumnIdentifyProperties(String indexPositionDelimiter, Integer indexPosition, String valueEndWithString) {
		this.indexPositionDelimiter = indexPositionDelimiter;
		this.indexPosition = indexPosition;
		this.valueEndWithString = valueEndWithString;
	}

	public String getIndexPositionDelimiter() {
		return indexPositionDelimiter;
	}

	public void setIndexPositionDelimiter(String indexPositionDelimiter) {
		this.indexPositionDelimiter = indexPositionDelimiter;
	}

	public Integer getIndexPosition() {
		return indexPosition;
	}

	public void setIndexPosition(Integer indexPosition) {
		this.indexPosition = indexPosition;
	}

	public String getValueEndWithString() {
		return valueEndWithString;
	}

	public void setValueEndWithString(String valueEndWithString) {
		this.valueEndWithString = valueEndWithString;
	}
}
