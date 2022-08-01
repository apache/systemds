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

import { NgModule, } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { LayoutComponent } from './modules/layout/layout.component';
import { DashboardComponent } from './modules/dashboard/main/dashboard.component';

import { HttpClientModule } from '@angular/common/http';

import { MaterialModule } from './material.module'

import { DragDropModule } from '@angular/cdk/drag-drop';
import { CoordinatorComponent } from './modules/dashboard/coordinator/coordinator.component';
import { ConnectionComponent } from './modules/dashboard/connection/connection.component';
import { WorkerComponent } from './modules/dashboard/worker/worker.component';
import { DialogDashboardComponent } from './modules/dashboard/dialog-dashboard/dialog-dashboard.component';
import { DashboardDirective } from './modules/dashboard/main/dashboard.directive';
import { ListCoordinatorsComponent } from './modules/coordinators/list/list.component';
import { ListWorkersComponent } from './modules/workers/list/list.component';
import { ViewWorkerComponent } from './modules/workers/view/view.component';
import { CreateEditCoordinatorsComponent } from "./modules/coordinators/create-edit/create-edit.component";
import { CreateEditWorkersComponent } from "./modules/workers/create-edit/create-edit.component";
import { ListWorkersEventsComponent } from "./modules/events/list/list.component";
import { ViewWorkerEventsComponent } from "./modules/events/view/view.component";

@NgModule({
	declarations: [
		AppComponent,
		LayoutComponent,
		DashboardComponent,
		CoordinatorComponent,
		ConnectionComponent,
		WorkerComponent,
		DialogDashboardComponent,
		DashboardDirective,
		ListCoordinatorsComponent,
		ListWorkersComponent,
		ViewWorkerComponent,
		CreateEditCoordinatorsComponent,
		CreateEditWorkersComponent,
		ListWorkersEventsComponent,
		ViewWorkerEventsComponent
	],
	imports: [
		BrowserModule,
		HttpClientModule,
		AppRoutingModule,
		BrowserAnimationsModule,
		MaterialModule,
		DragDropModule,
		FormsModule,
		ReactiveFormsModule,
	],
	providers: [],
	bootstrap: [AppComponent]
})
export class AppModule {
}
