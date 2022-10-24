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

import { ComponentFixture, TestBed } from '@angular/core/testing';

import { WorkerComponent } from './worker.component';
import { FederatedSiteService } from "../../../services/federatedSiteService.service";
import { FederatedSiteServiceStub } from "../../../services/federatedSiteService.stub";
import { DebugElement } from "@angular/core";
import { By } from "@angular/platform-browser";

describe('WorkerComponent', () => {
	let component: WorkerComponent;
	let fixture: ComponentFixture<WorkerComponent>;
	let de: DebugElement;

	beforeEach(async () => {
		await TestBed.configureTestingModule({
			declarations: [WorkerComponent],
			providers: [ { provide: FederatedSiteService , useClass: FederatedSiteServiceStub } ]
		})
		.compileComponents();
	});

	beforeEach(() => {
		fixture = TestBed.createComponent(WorkerComponent);
		component = fixture.componentInstance;
		de = fixture.debugElement;

		fixture.detectChanges();
	});

	// TODO: Add test integrated with dashboard component, since creation of worker component depends on it
});
