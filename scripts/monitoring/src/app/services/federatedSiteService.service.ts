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

import { Injectable } from '@angular/core';
import { Observable, retry, share, Subject, switchMap, takeUntil, timer } from 'rxjs';
import { constants } from '../constants';
import { Coordinator } from '../models/coordinator.model';
import { Worker } from '../models/worker.model';
import { HttpClient } from "@angular/common/http";
import { Statistics } from "../models/statistics.model";

@Injectable({
	providedIn: 'root'
})
export class FederatedSiteService {

	constructor(private http: HttpClient) { }

	public getAllCoordinators(): Observable<Coordinator[]> {
		return this.http.get<Coordinator[]>(constants.uriParts.coordinators);
	}

	public getAllWorkers(): Observable<Worker[]> {
		return this.http.get<Worker[]>(constants.uriParts.workers);
	}

	public getCoordinator(id: number): Observable<Coordinator> {
		return this.http.get<Coordinator>(constants.uriParts.coordinators + "/" + id.toString());
	}

	public getWorker(id: number): Observable<Worker> {
		return this.http.get<Worker>(constants.uriParts.workers + "/" + id.toString());
	}

	public getWorkerPolling(id: number, stopPolling: Subject<any>): Observable<Worker> {
		return timer(1, 3000).pipe(
			switchMap(() => this.getWorker(id)),
			retry(),
			share(),
			takeUntil(stopPolling)
		);
	}

	public createCoordinator(coordinator: Coordinator): Observable<Coordinator> {
		let coordinatorModel = (({name, host, processId}) => ({name, host, processId}))(coordinator);

		return this.http.post<Coordinator>(constants.uriParts.coordinators, coordinatorModel);
	}

	public createWorker(worker: Worker): Observable<Worker> {
		let workerModel = (({name, address}) => ({name, address}))(worker);

		return this.http.post<Worker>(constants.uriParts.workers, workerModel);
	}

	public editCoordinator(coordinator: Coordinator): Observable<Coordinator> {
		let coordinatorModel = (({id, name, host, processId}) => ({id, name, host, processId}))(coordinator);

		return this.http.put<Coordinator>(constants.uriParts.coordinators + "/" + coordinator.id.toString(), coordinatorModel);
	}

	public editWorker(worker: Worker): Observable<Worker> {
		let workerModel = (({id, name, address}) => ({id, name, address}))(worker);

		return this.http.put<Worker>(constants.uriParts.workers + "/" + worker.id.toString(), workerModel);
	}

	public deleteCoordinator(id: number): Observable<Object> {
		return this.http.delete(constants.uriParts.coordinators + "/" + id.toString());
	}

	public deleteWorker(id: number): Observable<Object> {
		return this.http.delete(constants.uriParts.workers + "/" + id.toString());
	}

	public getStatistics(workerId: number): Observable<Statistics> {
		return this.http.get<Statistics>(constants.uriParts.statistics + "/" + workerId.toString());
	}

	public getStatisticsPolling(workerId: number, stopPolling: Subject<any>): Observable<Statistics> {
		return timer(1, 3000).pipe(
			switchMap(() => this.getStatistics(workerId)),
		retry(),
		share(),
		takeUntil(stopPolling)
		);
	}
}
