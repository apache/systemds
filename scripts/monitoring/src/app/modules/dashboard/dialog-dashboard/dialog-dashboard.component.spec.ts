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

import { DialogDashboardComponent } from './dialog-dashboard.component';
import { MAT_DIALOG_DATA, MatDialogRef } from "@angular/material/dialog";
import { By } from "@angular/platform-browser";
import { DebugElement } from "@angular/core";

describe('DialogDashboardComponent', () => {
	let component: DialogDashboardComponent;
	let fixture: ComponentFixture<DialogDashboardComponent>;
	let de: DebugElement;

	beforeEach(async () => {
		await TestBed.configureTestingModule({
			declarations: [DialogDashboardComponent],
			providers: [
				{ provide : MAT_DIALOG_DATA, useValue : {} },
				{ provide: MatDialogRef, useValue: {} }
			]
		})
		.compileComponents();
	});

	beforeEach(() => {
		fixture = TestBed.createComponent(DialogDashboardComponent);
		component = fixture.componentInstance;
		de = fixture.debugElement;

		fixture.detectChanges();
	});

	it('should create', () => {
		expect(component).toBeTruthy();
	});

	it('should contain coordinators and workers selection', () => {
		let html = de.query(By.css('#dashboard-dialog-content')).nativeElement.innerText;
		expect(html).toContain('Coordinators');
		expect(html).toContain('Workers');
	});
});
