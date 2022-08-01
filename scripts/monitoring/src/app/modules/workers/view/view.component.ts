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

import { Component, ViewChild } from '@angular/core';
import { MatPaginator } from '@angular/material/paginator';
import { MatSort } from '@angular/material/sort';
import { ActivatedRoute } from '@angular/router';
import { Worker } from 'src/app/models/worker.model';
import { FederatedSiteService } from 'src/app/services/federatedSiteService.service';
import { Statistics } from "../../../models/statistics.model";
import { MatTableDataSource } from "@angular/material/table";
import { DataObject } from "../../../models/dataObject.model";
import { Chart, registerables } from "chart.js";
import { constants } from "../../../constants";
import 'chartjs-adapter-moment';

@Component({
	selector: 'app-view-worker',
	templateUrl: './view.component.html',
	styleUrls: ['./view.component.scss']
})
export class ViewWorkerComponent {

	public displayedColumns: string[] = ['varName', 'dataType', 'valueType', 'size'];
	public dataSource: MatTableDataSource<DataObject> = new MatTableDataSource<DataObject>([]);

	public model: Worker;
	public statistics: Statistics;

	@ViewChild(MatPaginator) paginator: MatPaginator;
	@ViewChild(MatSort) sort: MatSort;

	private timer: any;

	constructor(
		private fedSiteService: FederatedSiteService,
		private router: ActivatedRoute) {
		Chart.register(...registerables);
	}

	ngOnInit(): void {
		const id = Number(this.router.snapshot.paramMap.get('id'));
		this.fedSiteService.getWorker(id).subscribe(worker => {
			this.model = worker;
		});

		this.statistics = new Statistics();

		const cpuMetricEle: any = document.getElementById('cpu-metric');
		const memoryMetricEle: any = document.getElementById('memory-metric');
		const requestsMetricEle: any = document.getElementById('requests-metric');

		let cpuChart = new Chart(cpuMetricEle.getContext('2d'), {
			type: 'line',
			data: {
				datasets: [{
					data: this.statistics.utilization.map(s => {
						return {x: s.timestamp, y: s.cpuUsage}
					}),
					borderColor: constants.chartColors.blue
				}]
			},
			options: {
				responsive: true,
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'CPU usage %'
					}
				},
				scales: {
					x: {
						type: 'time',
						time: {
							unit: 'second',
						}
					}
				}
			},
		});

		let memoryChart = new Chart(memoryMetricEle.getContext('2d'), {
			type: 'line',
			data: {
				datasets: [{
					data: this.statistics.utilization.map(s => {
						return {x: s.timestamp, y: s.memoryUsage}
					}),
					borderColor: constants.chartColors.red
				}]
			},
			options: {
				responsive: true,
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'Memory usage %'
					}
				},
				scales: {
					x: {
						type: 'time',
						time: {
							unit: 'second'
						}
					}
				}
			},
		});

		let requestsChart = new Chart(requestsMetricEle.getContext('2d'), {
			type: 'bar',
			data: {
				labels: this.statistics.requests.map(r => r.type),
				datasets: [{
					label: 'My First Dataset',
					data: this.statistics.requests.map(r => r.count),
					backgroundColor: constants.chartColors.purple,
				}]
			},
			options: {
				responsive: true,
				plugins: {
					legend: {
						display: false
					},
					title: {
						display: true,
						text: 'Request type count'
					}
				},
				scales: {
					y: {
						beginAtZero: true
					}
				}
			},
		});

		this.timer = setInterval(() => {

			this.fedSiteService.getWorker(id).subscribe(worker => this.model = worker);

			this.fedSiteService.getStatistics(id).subscribe(stats => {
				this.statistics = stats;

				cpuChart.data.datasets.forEach((dataset) => {
					dataset.data = [];
					this.statistics.utilization.map(s => dataset.data.push({ x: s.timestamp, y: s.cpuUsage }));
				});

				memoryChart.data.datasets.forEach((dataset) => {
					dataset.data = [];
					this.statistics.utilization.map(s => dataset.data.push({ x: s.timestamp, y: s.memoryUsage }));
				});

				requestsChart.data.labels = this.statistics.requests.map(r => r.type);
				requestsChart.data.datasets.forEach((dataset) => {
					dataset.data = [];
					this.statistics.requests.map(s => dataset.data.push(s.count));
				});

				cpuChart.update();
				memoryChart.update();
				requestsChart.update();

				this.dataSource.data = this.statistics.dataObjects;
			})
		}, 3000);

	}

	ngOnDestroy() {
		clearInterval(this.timer);
	}

}
