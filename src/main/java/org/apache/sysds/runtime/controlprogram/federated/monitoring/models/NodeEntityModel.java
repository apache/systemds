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
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;

import java.util.ArrayList;
import java.util.List;

public class NodeEntityModel extends BaseEntityModel {
	private Long _id;
	private String _name;
	private String _address;
	private boolean _isOnline = false;
	private double _jitCompileTime;

	private List<Pair<FederatedRequest.RequestType, Long>> _requestTypeCount;
	private List<BaseEntityModel> _stats;

	public NodeEntityModel() { }

	public NodeEntityModel(final Long id, final String name, final String address) {
		_id = id;
		_name = name;
		_address = address;
	}

	public Long getId() {
		return _id;
	}

	public void setId(final Long id) {
		_id = id;
	}

	public Boolean getOnlineStatus() {
		return _isOnline;
	}

	public void setOnlineStatus(final Boolean isOnline) {
		_isOnline = isOnline;
	}

	public String getName() {
		return _name;
	}

	public void setName(final String name) {
		_name = name;
	}

	public String getAddress() {
		return _address;
	}

	public void setAddress(final String address) {
		_address = address;
	}

	public List<BaseEntityModel> getStats() {
		return _stats;
	}

	public void setStats(final List<BaseEntityModel> stats) {
		_stats = stats;
	}

	public double getJitCompileTime() {
		return _jitCompileTime;
	}

	public void setJitCompileTime(final double jitCompileTime) {
		_jitCompileTime = jitCompileTime;
	}

	public String getRequestTypeCount() {
		List<String> requestTypeCountStrArr = new ArrayList<>();

		if (_requestTypeCount != null) {
			for(var entry : _requestTypeCount) {
				requestTypeCountStrArr.add(String.format("{" +
						"\"type\": \"%s\"," +
						"\"count\": %d" +
						"}", entry.getLeft().toString(), entry.getRight()));
			}
		}

		return String.join(",", requestTypeCountStrArr);
	}

	public void setRequestTypeCount(final List<Pair<FederatedRequest.RequestType, Long>> requestTypeCount) {
		_requestTypeCount = requestTypeCount;
	}

	@Override
	public String toString() {
		return String.format("{" +
				"\"id\": %d," +
				"\"name\": \"%s\"," +
				"\"address\": \"%s\"," +
				"\"isOnline\": %b," +
				"\"jitCompileTime\": %.2f," +
				"\"requestTypeCounts\": [%s]," +
				"\"stats\": %s" +
				"}", _id, _name, _address, _isOnline, _jitCompileTime, getRequestTypeCount(), _stats);
	}
}
