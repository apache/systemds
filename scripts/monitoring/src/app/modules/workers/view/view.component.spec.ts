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

import { ViewWorkerComponent } from './view.component';
import { FederatedSiteService } from "../../../services/federatedSiteService.service";
import { FederatedSiteServiceStub } from "../../../services/federatedSiteService.stub";
import { ActivatedRoute } from "@angular/router";
import { DebugElement } from "@angular/core";
import { By } from "@angular/platform-browser";

describe('ViewWorkerComponent', () => {
	let component: ViewWorkerComponent;
	let fixture: ComponentFixture<ViewWorkerComponent>;
	let de: DebugElement;

	beforeEach(async () => {
		await TestBed.configureTestingModule({
			declarations: [ViewWorkerComponent],
			providers: [
				{ provide: FederatedSiteService , useClass: FederatedSiteServiceStub },
				{
					provide : ActivatedRoute,
					useValue : {
						snapshot: {
							paramMap: {
								get: () => {
									return { id: 1 }
								}
							}
						}
					}
				}
			]
		})
		.compileComponents();
	});

	beforeEach(() => {
		fixture = TestBed.createComponent(ViewWorkerComponent);
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

	it('should contain address and status information', () => {
		let html = de.query(By.css('#main-worker-information')).nativeElement.innerText;
		expect(html).toContain('Address');
		expect(html).toContain('Status');
	});

	it('should contain CPU metrics diagram', () => {
		expect(de.query(By.css('#cpu-metric-card'))).not.toBeNull();
	});

	it('should contain memory metrics diagram', () => {
		expect(de.query(By.css('#memory-metric-card'))).not.toBeNull();
	});
});
