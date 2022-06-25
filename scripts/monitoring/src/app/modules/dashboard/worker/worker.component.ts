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

import { Component } from '@angular/core';
import { Worker } from 'src/app/models/worker.model';
import { FederatedSiteService } from "../../../services/federatedSiteService.service";
import { ActivatedRoute } from "@angular/router";

@Component({
	selector: 'app-worker',
	templateUrl: './worker.component.html',
	styleUrls: ['./worker.component.scss']
})
export class WorkerComponent {

	public model: Worker;

	public optionsMemory: any;
	public updateOptionsMemory: any;
	public displayedColumns: string[] = ['type', 'time', 'frequency'];
	dataSource = [
		{type: 'fed_-', time: '0.417', frequency: 3},
		{type: 'fed_uamin', time: '0.156', frequency: 2},
		{type: 'JVM GC', time: '0.062', frequency: 1},
	];
	private dataMemory!: any[];
	private timer: any;

	constructor(private fedSiteService: FederatedSiteService) {	}

	ngOnInit(): void {
		this.model = this.model ? this.model : new Worker(-1, '', '', false, 0, [], []);

		this.fedSiteService.getWorker(this.model.id).subscribe(worker => {
			this.model = worker;

			this.updateMetrics();
		});

		this.dataMemory = [];

		this.optionsMemory = {
			title: {
				text: 'Memory (%)'
			},
			tooltip: {
				trigger: 'axis',
				formatter: (params: any) => {
					params = params[0];
					const date = new Date(params.name);
					return date.getDate() + '/' + (date.getMonth() + 1) + '/' + date.getFullYear() + ' : ' + params.value[1];
				},
				axisPointer: {
					animation: false
				}
			},
			xAxis: {
				type: 'time',
				splitLine: {
					show: false
				},
				show: false
			},
			yAxis: {
				type: 'value',
				boundaryGap: [0, '100%'],
				splitLine: {
					show: false
				}
			},
			series: [{
				name: 'Mocking Data',
				type: 'line',
				showSymbol: false,
				hoverAnimation: false,
				areaStyle: {},
				data: this.dataMemory
			}]
		};

		this.timer = setInterval(() => {
			this.fedSiteService.getWorker(this.model.id).subscribe(worker => {
				this.model = worker;

				this.updateMetrics();
			})
		}, 3000);
	}

	ngOnDestroy() {
		clearInterval(this.timer);
	}

	private updateMetrics(): void {

		console.log(this.model.stats);

		this.dataMemory = this.model.stats.map(s => {
			return {
				name: s.timestamp,
				value: [
					s.timestamp,
					s.memoryUsage
				]
			}
		})

		// update series data:
		this.updateOptionsMemory = {
			series: [{
				data: this.dataMemory
			}]
		};
	}
}
