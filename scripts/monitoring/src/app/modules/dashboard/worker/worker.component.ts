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
import { Statistics } from "../../../models/statistics.model";
import { MatTableDataSource } from "@angular/material/table";
import { Chart, registerables } from "chart.js";
import { constants } from "../../../constants";
import 'chartjs-adapter-moment';
import { Subject } from "rxjs";
import { Utils } from "../../../utils";

@Component({
	selector: 'app-worker',
	templateUrl: './worker.component.html',
	styleUrls: ['./worker.component.scss']
})
export class WorkerComponent {

	public workerId: number;

	public model: Worker;
	public statistics: Statistics;

	public displayedColumns: string[] = ['instruction', 'time', 'frequency'];
	public dataSource: MatTableDataSource<any> = new MatTableDataSource<any>([]);

	public heavyHittersCount: number = 3;
	public additionalCardHeight: number = 0;

	private stopPollingWorker = new Subject<any>();
	private stopPollingStatistics = new Subject<any>();

	constructor(private fedSiteService: FederatedSiteService) {
		Chart.register(...registerables);
	}

	ngOnInit(): void {
		this.statistics = new Statistics();

		const memoryMetricEle: any = document.querySelector(`#${constants.prefixes.worker + this.workerId} canvas`);

		let memoryChart = new Chart(memoryMetricEle.getContext('2d'), {
			type: 'line',
			data: {
				datasets: [{
					data: this.statistics.utilization.map(s => {
						return { x: s.timestamp, y: s.memoryUsage }
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
						grid: {
							display: false
						},
						type: 'timeseries',
						ticks: {
							display: false
						}
					},
					y: {
						beginAtZero: true
					}
				}
			},
		});

		this.fedSiteService.getWorkerPolling(this.workerId, this.stopPollingWorker).subscribe(worker => this.model = worker);

		this.fedSiteService.getStatisticsPolling(this.workerId, this.stopPollingStatistics).subscribe(stats => {
			this.statistics = stats;

			memoryChart.data.datasets.forEach((dataset) => {
				dataset.data = [];
				this.statistics.utilization.map(s => dataset.data.push({ x: s.timestamp, y: s.memoryUsage }));
				dataset.data.sort(Utils.sortTimestamp);
			});

			memoryChart.update();

			this.dataSource = this.parseInstructions();
		});
	}

	private parseInstructions(): any {
		let result: any = this.statistics.heavyHitters.map(hh => {
			return {
				instruction: hh.operation,
				time: hh.duration,
				frequency: hh.count
			}
		});

		// 48 px is the height of one table row
		this.additionalCardHeight = this.heavyHittersCount * 48;

		if (result.length < this.heavyHittersCount) {
			this.heavyHittersCount = result.length;
		}

		return result.sort((a,b) => b['time']-a['time']).slice(0,this.heavyHittersCount);
	}

	ngOnDestroy() {
		this.stopPollingWorker.next(null);
		this.stopPollingStatistics.next(null);
	}
}
