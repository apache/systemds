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

let coordinators = [
	{
		"id": 1,
		"name": "Coordinator 1",
		"address": "localhost:8445"
	},
	{
		"id": 2,
		"name": "Coordinator 1",
		"address": "localhost:8446"
	}
]

let workers = [
	{
		"id": 1,
		"name": "Worker 1",
		"address": "localhost:8001",
		"isOnline": false,
		"jitCompileTime": 0,
		"requestTypeCounts": [
			{"type": "GET_VAR", "count": 0},
			{"type": "PUT_VAR", "count": 0},
			{"type": "READ_VAR", "count": 0},
			{"type": "EXEC_UDF", "count": 0},
			{"type": "EXEC_INST", "count": 0}
		],
		"stats": []
	},
	{
		"id": 2,
		"name": "Worker 2",
		"address": "localhost:8002",
		"isOnline": false,
		"jitCompileTime": 0,
		"requestTypeCounts": [
			{"type": "GET_VAR", "count": 0},
			{"type": "PUT_VAR", "count": 0},
			{"type": "READ_VAR", "count": 0},
			{"type": "EXEC_UDF", "count": 0},
			{"type": "EXEC_INST", "count": 0}
		],
		"stats": [{
			"timestamp": "2022-06-25 13:18:19.578",
			"x": 1.96,
			"memoryUsage": 1.46,
			"coordinatorTraffic": [],
			"heavyHitters": []
		}, {
			"timestamp": "2022-06-25 13:18:22.522",
			"x": 1.95,
			"memoryUsage": 1.46,
			"coordinatorTraffic": [],
			"heavyHitters": []
		}]
	},
	{
		"id": 3,
		"name": "Worker 3",
		"address": "localhost:8003",
		"isOnline": true,
		"jitCompileTime": 3.69,
		"requestTypeCounts": [
			{"type": "GET_VAR", "count": 0},
			{"type": "PUT_VAR", "count": 0},
			{"type": "READ_VAR", "count": 0},
			{"type": "EXEC_UDF", "count": 0},
			{"type": "EXEC_INST", "count": 0}
		],
		"stats": [{
			"timestamp": "2022-06-25 13:18:19.578",
			"x": 1.96,
			"memoryUsage": 1.46,
			"coordinatorTraffic": [],
			"heavyHitters": []
		}, {
			"timestamp": "2022-06-25 13:18:22.522",
			"x": 1.95,
			"memoryUsage": 1.46,
			"coordinatorTraffic": [],
			"heavyHitters": []
		}]
	}
]

export const serviceMockData = {
	workers: workers,
	coordinators: coordinators
}
