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

import { Utilization } from "./utilization.model";
import { Traffic } from "./traffic.model";
import { Event } from "./event.model";
import { DataObject } from "./dataObject.model";
import { FedRequest } from "./fedRequest.model";
import { HeavyHitter } from "./heavyHitter.model";

export class Statistics {
	constructor(public utilization: Utilization[] = [],
				public traffic: Traffic[] = [],
				public events: Event[] = [],
				public dataObjects: DataObject[] = [],
				public requests: FedRequest[] = [],
				public heavyHitters: HeavyHitter[] = []) { }
}
