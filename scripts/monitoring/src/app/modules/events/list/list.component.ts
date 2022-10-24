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

import { Component, ViewChild } from '@angular/core';
import { MatSort } from '@angular/material/sort';
import { Router } from '@angular/router';

import { Worker } from 'src/app/models/worker.model';
import { FederatedSiteService } from 'src/app/services/federatedSiteService.service';
import { MatTableDataSource } from "@angular/material/table";
import { MatDialog } from "@angular/material/dialog";

@Component({
	selector: 'app-list-workers-events',
	templateUrl: './list.component.html',
	styleUrls: ['./list.component.scss']
})
export class ListWorkersEventsComponent {

	public displayedColumns: string[] = ['name', 'address', 'status', 'actions'];
	public dataSource: MatTableDataSource<Worker> = new MatTableDataSource<Worker>([]);

	public loadingData: boolean = false;

	@ViewChild(MatSort, {static: true})
	sort: MatSort = new MatSort;

	constructor(
		public dialog: MatDialog,
		private fedSiteService: FederatedSiteService,
		private router: Router) {
	}

	ngOnInit(): void {
		this.refreshData();
	}

	viewEvent(workerId: number) {
		this.router.navigate(['/events/' + workerId])
	}

	refreshData() {
		this.loadingData = true;
		this.fedSiteService.getAllWorkers().subscribe(workers => {
			this.dataSource.data = workers
			this.loadingData = false;
		});
	}

}
