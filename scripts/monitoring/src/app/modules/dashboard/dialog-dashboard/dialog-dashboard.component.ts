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

import { Component, Inject } from '@angular/core';
import { DashboardComponent } from '../main/dashboard.component';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { FedSiteData } from 'src/app/models/fedSiteData.model';

@Component({
	selector: 'app-dialog-dashboard',
	templateUrl: './dialog-dashboard.component.html',
	styleUrls: ['./dialog-dashboard.component.scss']
})
export class DialogDashboardComponent {

	private selectedCoordinatorIds: number[] = [];
	private selectedWorkerIds: number[] = [];

	constructor(
		public dialogRef: MatDialogRef<DashboardComponent>,
		@Inject(MAT_DIALOG_DATA) public data: FedSiteData
	) {
	}

	changeSelectedCoordinators(id: number): void {
		if (this.selectedCoordinatorIds.some(c => c === id)) {
			this.selectedCoordinatorIds = this.selectedCoordinatorIds.filter(c => c !== id);
		} else {
			this.selectedCoordinatorIds.push(id);
		}
	}

	changeSelectedWorkers(id: number): void {
		if (this.selectedWorkerIds.some(w => w === id)) {
			this.selectedWorkerIds = this.selectedWorkerIds.filter(w => w !== id);
		} else {
			this.selectedWorkerIds.push(id);
		}
	}

	onSaveClick(): void {

		this.dialogRef.close({
			selectedWorkerIds: this.selectedWorkerIds,
			selectedCoordinatorIds: this.selectedCoordinatorIds
		});
	}

	onCancelClick(): void {
		this.dialogRef.close();
	}
}
