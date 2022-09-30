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

import { Observable, of } from 'rxjs';
import { Coordinator } from '../models/coordinator.model';
import { Worker } from '../models/worker.model';
import { serviceMockData } from "./service-mock-data";
import { constants } from "../constants";

export class FederatedSiteServiceStub {

	public getAllCoordinators() {
		return of(serviceMockData.coordinators);
	}

	public loadCoordinators() {
		return of(serviceMockData.coordinators);
	}

	public loadWorkers() {
		return of(serviceMockData.workers);
	}

	public getAllWorkers() {
		return of(serviceMockData.workers)
	}

	public getCoordinator(id: number){
		return of(serviceMockData.coordinators.find(c => c.id === id));
	}

	public getWorker(id: number) {
		return of(serviceMockData.workers.find(w => w.id === id));
	}

	public getWorkerPolling(id: number) {
		return of(serviceMockData.workers.find(w => w.id === id));
	}

	public getStatisticsPolling(id: number) {
		return of(serviceMockData.statistics);
	}

	public createCoordinator(coordinator: Coordinator) {
		return of(coordinator)
	}

	public createWorker(worker: Worker) {
		return of(worker);
	}

	public editCoordinator(coordinator: Coordinator) {
		return of({"id": 42});
	}

	public editWorker(worker: Worker) {
		return of({"id": 42});
	}

	public deleteCoordinator(id: number) {
		return of({"id": id});
	}

	public deleteWorker(id: number) {
		return of({"id": id});
	}
}
