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

import { LayoutComponent } from './layout.component';
import { MatDialog } from "@angular/material/dialog";
import { ChangeDetectorRef, DebugElement } from "@angular/core";
import { MediaMatcher } from "@angular/cdk/layout";
import { By } from "@angular/platform-browser";

describe('LayoutComponent', () => {
	let component: LayoutComponent;
	let fixture: ComponentFixture<LayoutComponent>;
	let de: DebugElement;

	beforeEach(async () => {
		await TestBed.configureTestingModule({
			declarations: [LayoutComponent],
			providers: [
				{ provide: MatDialog, useValue: {} },
				{ provide: ChangeDetectorRef, useValue: {} },
				{
					provide: MediaMatcher,
					useValue: {
						matchMedia: () => {
							return {
								addListener: () => {},
								removeListener: () => {}
							}
						}
					}
				}
			]
		})
		.compileComponents();
	});

	beforeEach(() => {
		fixture = TestBed.createComponent(LayoutComponent);
		component = fixture.componentInstance;
		de = fixture.debugElement;

		fixture.detectChanges();
	});

	it('should create', () => {
		expect(component).toBeTruthy();
	});

	it('should contain dashboard, coordinators and workers menu elements', () => {
		let html = de.query(By.css('#menu-elements')).nativeElement.innerText;

		expect(html).not.toBeNull();
		expect(html).toContain('Dashboard');
		expect(html).toContain('Coordinators');
		expect(html).toContain('Workers');
	});

	it('should contain register entity button', () => {
		expect(de.query(By.css('#register-entity'))).not.toBeNull();
	});
});
