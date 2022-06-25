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

public class Constants {
	public static final String REQUEST_TYPE_COUNT_JSON_STR = "{ \"type\": \"%s\", \"count\": %d }";
	public static final String NODE_ENTITY_JSON_STR = "{" +
			"\"id\": %d," +
			"\"name\": \"%s\"," +
			"\"address\": \"%s\"," +
			"\"isOnline\": %b," +
			"\"jitCompileTime\": %.2f," +
			"\"requestTypeCounts\": [%s]," +
			"\"stats\": %s" +
		"}";
	public static final String STATS_ENTITY_JSON_STR = "{" +
			"\"timestamp\": \"%s\"," +
			"\"cpuUsage\": %.2f," +
			"\"memoryUsage\": %.2f," +
			"\"coordinatorTraffic\": [%s]," +
			"\"heavyHitters\": [%s]" +
		"}";
	public static final String TRANSFERRED_BYTES_JSON_STR = "{" +
			"\"datetime\": \"%s\"," +
			"\"coordinatorAddress\": \"%s\"," +
			"\"byteAmount\": %d" +
		"}";
	public static final String HEAVY_HITTER_INSTRUCTIONS_JSON_STR = "{" +
			"\"instruction\": \"%s\"," +
			"\"count\": %d," +
			"\"duration\": %.2f" +
		"}";
}