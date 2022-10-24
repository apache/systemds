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

import java.util.List;
import java.util.stream.Collectors;

public class StatisticsModel extends BaseModel {
	private static final long serialVersionUID = -2492467768854934429L;
	public List<UtilizationModel> utilization;
	public List<TrafficModel> traffic;
	public List<EventModel> events;
	public List<DataObjectModel> dataObjects;
	public List<RequestModel> requests;
	public List<HeavyHitterModel> heavyHitters;

	private static final String JsonFormat = "{" +
			"\"utilization\": [%s]," +
			"\"traffic\": [%s]," +
			"\"events\": [%s]," +
			"\"dataObjects\": [%s]," +
			"\"requests\": [%s]," +
			"\"heavyHitters\": [%s]" +
			"}";

	public StatisticsModel() { }

	public StatisticsModel(List<UtilizationModel> utilization,
						   List<TrafficModel> traffic,
						   List<EventModel> events,
						   List<DataObjectModel> dataObjects,
						   List<RequestModel> requests,
						   List<HeavyHitterModel> heavyHitters) {
		this.utilization = utilization;
		this.traffic = traffic;
		this.events = events;
		this.dataObjects = dataObjects;
		this.requests = requests;
		this.heavyHitters = heavyHitters;
	}


	@Override
	public String toString() {
		String utilizationStr = null, trafficStr = null, eventsStr = null, dataObjectsStr = null, requestsStr = null, heavyHittersStr = null;

		if (utilization != null) {
			utilizationStr = utilization.stream()
					.map(UtilizationModel::toString)
					.collect(Collectors.joining(","));
		}

		if (traffic != null) {
			trafficStr = traffic.stream()
					.map(TrafficModel::toString)
					.collect(Collectors.joining(","));
		}

		if (events != null) {
			eventsStr = events.stream()
					.map(EventModel::toString)
					.collect(Collectors.joining(","));
		}

		if (dataObjects != null) {
			dataObjectsStr = dataObjects.stream()
					.map(DataObjectModel::toString)
					.collect(Collectors.joining(","));
		}

		if (requests != null) {
			requestsStr = requests.stream()
					.map(RequestModel::toString)
					.collect(Collectors.joining(","));
		}

		if (heavyHitters != null) {
			heavyHittersStr = heavyHitters.stream()
					.map(HeavyHitterModel::toString)
					.collect(Collectors.joining(","));
		}

		return String.format(JsonFormat, utilizationStr, trafficStr, eventsStr, dataObjectsStr, requestsStr, heavyHittersStr);
	}
}
