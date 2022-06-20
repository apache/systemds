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

package org.apache.sysds.runtime.controlprogram.federated.monitoring.repositories;

public class Constants {
	public static final String WORKERS_TABLE_NAME= "workers";
	public static final String COORDINATORS_TABLE_NAME= "coordinators";
	public static final String STATS_TABLE_NAME= "statistics";
	public static final String ENTITY_NAME_COL = "name";
	public static final String ENTITY_ADDR_COL = "address";
	public static final String ENTITY_CPU_COL = "cpuUsage";
	public static final String ENTITY_MEM_COL = "memoryUsage";
	public static final String ENTITY_TRAFFIC_COL = "coordinatorTraffic";
	public static final String ENTITY_HEAVY_HITTERS_COL = "heavyHitters";
	public static final String ENTITY_ID_COL = "id";
	public static final String ENTITY_WORKER_ID_COL = "workerId";
}
