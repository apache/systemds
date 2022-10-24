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

import { CreateEditCoordinatorsComponent } from './create-edit.component';
import { DebugElement } from "@angular/core";
import { By } from "@angular/platform-browser";
import { FederatedSiteService } from "../../../services/federatedSiteService.service";
import { FederatedSiteServiceStub } from "../../../services/federatedSiteService.stub";
import { MAT_DIALOG_DATA, MatDialogRef } from "@angular/material/dialog";

describe('CreateEditCoordinatorsComponent', () => {
	let component: CreateEditCoordinatorsComponent;
	let fixture: ComponentFixture<CreateEditCoordinatorsComponent>;
	let de: DebugElement;

	beforeEach(async () => {
		await TestBed.configureTestingModule({
			declarations: [ CreateEditCoordinatorsComponent ],
			providers: [
				{ provide: FederatedSiteService , useClass: FederatedSiteServiceStub },
				{ provide : MAT_DIALOG_DATA, useValue : {} },
				{ provide: MatDialogRef, useValue: {} }
			]
		})
		.compileComponents();
	});

	beforeEach(() => {
		fixture = TestBed.createComponent(CreateEditCoordinatorsComponent);
		component = fixture.componentInstance;
		de = fixture.debugElement;

		fixture.detectChanges();
	});

	it('should create', () => {
		expect(component).toBeTruthy();
	});

	it('should not contain null model', () => {
		expect(component.model).not.toBeNull();
	});

	it('should contain name host and processId fields', () => {
		let html = de.query(By.css('#register-coordinator-content')).nativeElement.innerText;
		expect(html).toContain('Name');
		expect(html).toContain('Host');
		expect(html).toContain('Process Id');
	});
});
