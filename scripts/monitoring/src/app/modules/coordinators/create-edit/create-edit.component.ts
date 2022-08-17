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
import { Coordinator } from "../../../models/coordinator.model";
import { MAT_DIALOG_DATA, MatDialogRef } from "@angular/material/dialog";
import { FederatedSiteService } from "../../../services/federatedSiteService.service";

@Component({
	selector: 'app-create-edit-coordinator',
	templateUrl: './create-edit.component.html',
	styleUrls: ['./create-edit.component.scss']
})
export class CreateEditCoordinatorsComponent {

	public model: Coordinator;

	constructor(
		private fedSiteService: FederatedSiteService,
		public dialogRef: MatDialogRef<CreateEditCoordinatorsComponent>,
		@Inject(MAT_DIALOG_DATA) public id: number) {
	}

	ngOnInit(): void {
		this.model = new Coordinator();

		if (this.id !== null) {
			this.fedSiteService.getCoordinator(this.id).subscribe(coordinator => this.model = coordinator);
		}
	}

	onSaveClick() {

		if (this.id !== null) {
			this.fedSiteService.editCoordinator(this.model).subscribe(coordinator => this.model = coordinator);
		} else {
			this.fedSiteService.createCoordinator(this.model).subscribe(coordinator => this.model = coordinator);
		}

		this.dialogRef.close()
	}

	onCancelClick() {
		this.dialogRef.close()
	}
}
