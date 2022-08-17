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
import { Worker } from "../../../models/worker.model";
import { MAT_DIALOG_DATA, MatDialogRef } from "@angular/material/dialog";
import { FederatedSiteService } from "../../../services/federatedSiteService.service";

@Component({
	selector: 'app-create-edit-worker',
	templateUrl: './create-edit.component.html',
	styleUrls: ['./create-edit.component.scss']
})
export class CreateEditWorkersComponent {

	public model: Worker;

	constructor(
		private fedSiteService: FederatedSiteService,
		public dialogRef: MatDialogRef<CreateEditWorkersComponent>,
		@Inject(MAT_DIALOG_DATA) public id: number) {
	}

	ngOnInit(): void {
		this.model = new Worker();

		if (this.id !== null) {
			this.fedSiteService.getWorker(this.id).subscribe(worker => this.model = worker);
		}
	}

	onSaveClick() {

		if (this.id !== null) {
			this.fedSiteService.editWorker(this.model).subscribe(worker => {
				this.model = worker;
			});
		} else {
			this.fedSiteService.createWorker(this.model).subscribe(worker => {
				this.model = worker;
			});
		}

		this.dialogRef.close()
	}

	onCancelClick() {
		this.dialogRef.close()
	}
}
