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
import { BehaviorSubject, Observable } from 'rxjs';
import { constants } from '../constants';
import { Coordinator } from '../models/coordinator.model';
import { Worker } from '../models/worker.model';
import { HttpClient } from "@angular/common/http";

@Injectable({
	providedIn: 'root'
})
export class FederatedSiteService {

	private coordinators: BehaviorSubject<Coordinator[]> = new BehaviorSubject<Coordinator[]>([]);
	private workers: BehaviorSubject<Worker[]> = new BehaviorSubject<Worker[]>([]);

	constructor(private http: HttpClient) {
	}

	public getAllCoordinators(): Observable<Coordinator[]> {
		return this.coordinators.asObservable();
	}

	public getAllWorkers(): Observable<Worker[]> {
		return this.workers.asObservable();
	}

	public addCachedCoordinator(coordinator: Coordinator): void {
		let allCoordinators = this.coordinators.getValue();
		if (!allCoordinators.some(c => c.id === coordinator.id)) {
			allCoordinators.push(coordinator);
		} else {
			allCoordinators = allCoordinators.map(item => {
				item = item.id === coordinator.id ? coordinator : item
				return item;
			});
		}

		this.coordinators.next(allCoordinators);
	}

	public addCachedWorker(worker: Worker): void {
		let allWorkers = this.workers.getValue();
		if (!allWorkers.some(w => w.id === worker.id)) {
			allWorkers.push(worker);
		} else {
			allWorkers = allWorkers.map(item => {
				item = item.id === worker.id ? worker : item
				return item;
			});
		}

		this.workers.next(allWorkers);
	}

	public removeCachedCoordinator(id: number): void {
		let allCoordinators = this.coordinators.getValue().filter(c => c.id !== id);

		this.coordinators.next(allCoordinators);
	}

	public removeCachedWorker(id: number): void {
		let allWorkers = this.workers.getValue().filter(w => w.id !== id);

		this.workers.next(allWorkers);
	}

	public loadCoordinators(): Observable<Coordinator[]> {
		return this.http.get<Coordinator[]>(constants.uriParts.coordinators);
	}

	public loadWorkers(): Observable<Worker[]> {
		return this.http.get<Worker[]>(constants.uriParts.workers);
	}

	public getCoordinator(id: number): Observable<Coordinator> {
		return this.http.get<Coordinator>(constants.uriParts.coordinators + "/" + id.toString());
	}

	public getWorker(id: number): Observable<Worker> {
		return this.http.get<Worker>(constants.uriParts.workers + "/" + id.toString());
	}

	public createCoordinator(coordinator: Coordinator): Observable<Coordinator> {
		let coordinatorModel = (({name, address}) => ({name, address}))(coordinator);

		return this.http.post<Coordinator>(constants.uriParts.coordinators, coordinatorModel);
	}

	public createWorker(worker: Worker): Observable<Worker> {
		let workerModel = (({name, address}) => ({name, address}))(worker);

		return this.http.post<Worker>(constants.uriParts.workers, workerModel);
	}

	public editCoordinator(coordinator: Coordinator): Observable<Object> {
		let coordinatorModel = (({id, name, address}) => ({id, name, address}))(coordinator);

		return this.http.put(constants.uriParts.coordinators + "/" + coordinator.id.toString(), coordinatorModel);
	}

	public editWorker(worker: Worker): Observable<Object> {
		let workerModel = (({id, name, address}) => ({id, name, address}))(worker);

		return this.http.put(constants.uriParts.workers + "/" + worker.id.toString(), workerModel);
	}

	public deleteCoordinator(id: number): Observable<Object> {
		return this.http.delete(constants.uriParts.coordinators + "/" + id.toString());
	}

	public deleteWorker(id: number): Observable<Object> {
		return this.http.delete(constants.uriParts.workers + "/" + id.toString());
	}
}
