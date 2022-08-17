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

import { Component, OnInit, ChangeDetectorRef, OnDestroy, AfterViewInit } from '@angular/core';
import { MediaMatcher } from '@angular/cdk/layout';
import { MatDialog } from "@angular/material/dialog";
import { CreateEditCoordinatorsComponent } from "../coordinators/create-edit/create-edit.component";
import { CreateEditWorkersComponent } from "../workers/create-edit/create-edit.component";

@Component({
	selector: 'app-layout',
	templateUrl: './layout.component.html',
	styleUrls: ['./layout.component.scss']
})
export class LayoutComponent implements OnInit, OnDestroy, AfterViewInit {

	mobileQuery: MediaQueryList;
	private _mobileQueryListener: () => void;

	constructor(public dialog: MatDialog,
				private changeDetectorRef: ChangeDetectorRef,
				private media: MediaMatcher) {

		this.mobileQuery = this.media.matchMedia('(max-width: 1000px)');
		this._mobileQueryListener = () => changeDetectorRef.detectChanges();
		// tslint:disable-next-line: deprecation
		this.mobileQuery.addListener(this._mobileQueryListener);
	}

	ngOnInit(): void {
	}

	ngOnDestroy(): void {
		// tslint:disable-next-line: deprecation
		this.mobileQuery.removeListener(this._mobileQueryListener);
	}

	ngAfterViewInit(): void {
		this.changeDetectorRef.detectChanges();
	}

	openNewEntityDialog(type: 'worker' | 'coordinator'): void {

		if (type === 'worker') {
			this.dialog.open(CreateEditWorkersComponent, {
				width: '500px',
				data: null
			});
		} else {
			this.dialog.open(CreateEditCoordinatorsComponent, {
				width: '500px',
				data: null
			});
		}
	}
}
