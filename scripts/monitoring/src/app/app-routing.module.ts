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

import { NgModule } from "@angular/core";
import { RouterModule, Routes } from "@angular/router";
import { ListCoordinatorsComponent } from "./modules/coordinators/list/list.component";
import { DashboardComponent } from "./modules/dashboard/main/dashboard.component";
import { LayoutComponent } from "./modules/layout/layout.component";
import { ListWorkersComponent } from "./modules/workers/list/list.component";
import { ViewWorkerComponent } from "./modules/workers/view/view.component";
import { ListWorkersEventsComponent } from "./modules/events/list/list.component";
import { ViewWorkerEventsComponent } from "./modules/events/view/view.component";

const routes: Routes = [
	{
		path: '',
		component: LayoutComponent,
		children: [
			{
				path: 'dashboard',
				component: DashboardComponent
			},
			{
				path: 'coordinators',
				component: ListCoordinatorsComponent,
			},
			{
				path: 'workers',
				component: ListWorkersComponent,
			},
			{
				path: 'workers/:id',
				component: ViewWorkerComponent
			},
			{
				path: 'events',
				component: ListWorkersEventsComponent,
			},
			{
				path: 'events/:id',
				component: ViewWorkerEventsComponent
			},
			{path: '', redirectTo: 'dashboard', pathMatch: 'full'}
		]
	},
];

@NgModule({
	imports: [RouterModule.forRoot(routes, {useHash: true})],
	exports: [RouterModule]
})
export class AppRoutingModule {
}
