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

import { Coordinator } from 'src/app/models/coordinator.model';
import { FederatedSiteService } from 'src/app/services/federatedSiteService.service';
import { MatTableDataSource } from "@angular/material/table";
import { MatDialog } from "@angular/material/dialog";
import { CreateEditCoordinatorsComponent } from "../create-edit/create-edit.component";

@Component({
	selector: 'app-list-coordinators',
	templateUrl: './list.component.html',
	styleUrls: ['./list.component.scss']
})
export class ListCoordinatorsComponent {

	public displayedColumns: string[] = ['name', 'host', 'processId', 'actions'];
	public dataSource: MatTableDataSource<Coordinator> = new MatTableDataSource<Coordinator>([]);

	@ViewChild(MatSort, {static: true})
	sort: MatSort = new MatSort;

	public loadingData: boolean = false;

	constructor(
		public dialog: MatDialog,
		private fedSiteService: FederatedSiteService) {
	}

	ngOnInit(): void {
		this.refreshData();
	}

	editCoordinator(id: number) {
		this.dialog.open(CreateEditCoordinatorsComponent, {
			width: '500px',
			data: id
		});
	}

	deleteCoordinator(id: number) {
		this.fedSiteService.deleteCoordinator(id).subscribe(() => {
			this.dataSource.data = this.dataSource.data.filter(c => c.id !== id)
		});
	}

	refreshData() {
		this.loadingData = true;
		this.fedSiteService.getAllCoordinators().subscribe(coordinators => {
			this.dataSource.data = coordinators
			this.loadingData = false;
		});
	}

}
