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

import { Component, OnInit, Output, ViewChild, ViewEncapsulation } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';

import { jsPlumb, jsPlumbInstance } from 'jsplumb';
import { constants } from 'src/app/constants';
import { FederatedSiteService } from 'src/app/services/federatedSiteService.service';
import { CoordinatorComponent } from '../coordinator/coordinator.component';
import { DialogDashboardComponent } from '../dialog-dashboard/dialog-dashboard.component';
import { WorkerComponent } from '../worker/worker.component';
import { DashboardDirective } from './dashboard.directive';
import { FedSiteData } from "../../../models/fedSiteData.model";
import { Coordinator } from "../../../models/coordinator.model";
import { Worker } from "../../../models/worker.model";

@Component({
	selector: 'app-dashboard',
	templateUrl: './dashboard.component.html',
	styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {

	public fedSiteData: FedSiteData;
	@ViewChild(DashboardDirective, {static: true}) fedSiteHost!: DashboardDirective;
	private jsPlumbInstance: jsPlumbInstance;

	constructor(public dialog: MatDialog,
				private fedSiteService: FederatedSiteService) {
	}

	ngOnInit(): void {

		this.fedSiteData = {
			workers: [],
			coordinators: []
		};

		this.jsPlumbInstance = jsPlumb.getInstance();
		this.jsPlumbInstance.setContainer('dashboard-content');
	}

	openConfigDialog(): void {

		this.fedSiteService.loadCoordinators().subscribe(coordinators =>
			coordinators.forEach(coordinator => this.fedSiteService.addCachedCoordinator(coordinator)));
		this.fedSiteService.loadWorkers().subscribe(workers =>
			workers.forEach(worker => this.fedSiteService.addCachedWorker(worker)));

		this.fedSiteService.getAllCoordinators().subscribe(coordinators => this.fedSiteData.coordinators = coordinators);
		this.fedSiteService.getAllWorkers().subscribe(workers => this.fedSiteData.workers = workers);

		const dialogRef = this.dialog.open(DialogDashboardComponent, {
			width: '500px',
			data: this.fedSiteData,
		});

		dialogRef.afterClosed().subscribe(result => {
			if (result) {
				let selectedCoordinators = this.fedSiteData.coordinators.filter(c => result['selectedCoordinatorIds'].includes(c.id));
				let selectedWorkers = this.fedSiteData.workers.filter(w => result['selectedWorkerIds'].includes(w.id));

				this.fedSiteHost.viewContainerRef.clear();
				this.jsPlumbInstance.removeAllEndpoints('dashboard-content');

				this.redrawDiagram(selectedCoordinators, selectedWorkers);
			}
		});
	}

	private redrawDiagram(selectedCoordinators: Coordinator[], selectedWorkers: Worker[]) {

		for (const worker of selectedWorkers) {
			const workerComponentRef = this.fedSiteHost.viewContainerRef.createComponent(WorkerComponent);
			workerComponentRef.instance.model = worker;
			workerComponentRef.location.nativeElement.id = constants.prefixes.worker + worker.id;
			workerComponentRef.location.nativeElement.style = 'position: absolute;';

			this.jsPlumbInstance.draggable(constants.prefixes.worker + worker.id);
		}

		for (const coordinator of selectedCoordinators) {
			const coordinatorComponentRef = this.fedSiteHost.viewContainerRef.createComponent(CoordinatorComponent);
			coordinatorComponentRef.instance.model = coordinator;
			coordinatorComponentRef.location.nativeElement.id = constants.prefixes.coordinator + coordinator.id;
			coordinatorComponentRef.location.nativeElement.style = 'position: absolute;';

			this.jsPlumbInstance.draggable(constants.prefixes.coordinator + coordinator.id);

			// for (const childWorker of coordinator.workers) {
			//   if (!selectedWorkers.some(w => w.id == childWorker.id)) {
			//     continue;
			//   }
			//
			//   const connectionComponentRef = this.fedSiteHost.viewContainerRef.createComponent(ConnectionComponent);
			//   connectionComponentRef.location.nativeElement.id =
			//     constants.prefixes.coordinator + coordinator.id + constants.prefixes.worker + childWorker.id
			//
			//   this.jsPlumbInstance.connect({
			//     source: constants.prefixes.coordinator + coordinator.id,
			//     target: constants.prefixes.worker + childWorker.id,
			//     anchor: ['AutoDefault'],
			//     overlays: [
			//       ['Custom', {
			//         create: function(component: any) {
			//           return constants.prefixes.coordinator + coordinator.id + constants.prefixes.worker + childWorker.id;
			//         },
			//         location: 0.5,
			//       }]
			//     ]
			//   });
			// }
		}
	}
}
