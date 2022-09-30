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

import { DebugElement } from "@angular/core";
import { FederatedSiteService } from "../../../services/federatedSiteService.service";
import { FederatedSiteServiceStub } from "../../../services/federatedSiteService.stub";
import { Router } from "@angular/router";
import { MatDialog } from "@angular/material/dialog";
import { By } from "@angular/platform-browser";
import { ListWorkersComponent } from "../../workers/list/list.component";

describe('ListWorkersComponent', () => {
	let component: ListWorkersComponent;
	let fixture: ComponentFixture<ListWorkersComponent>;
	let de: DebugElement;

	beforeEach(async () => {
		await TestBed.configureTestingModule({
			declarations: [ListWorkersComponent],
			providers: [
				{ provide: FederatedSiteService , useClass: FederatedSiteServiceStub },
				{ provide : Router, useValue : {} },
				{ provide: MatDialog, useValue: {} }
			]
		})
		.compileComponents();
	});

	beforeEach(() => {
		fixture = TestBed.createComponent(ListWorkersComponent);
		component = fixture.componentInstance;
		de = fixture.debugElement;

		fixture.detectChanges();
	});

	it('should create', () => {
		expect(component).toBeTruthy();
	});

	it('should contain table of workers', () => {
		expect(de.query(By.css('table'))).not.toBeNull();
	});

	it('should contain name, address and actions table fields', () => {
		expect(component.displayedColumns).toContain('name');
		expect(component.displayedColumns).toContain('address');
		expect(component.displayedColumns).toContain('actions');
	});

	it('should not have null data source', () => {
		expect(component.dataSource).not.toBeNull();
	});
});
