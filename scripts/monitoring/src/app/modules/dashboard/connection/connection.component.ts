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

import { Component, OnInit } from '@angular/core';
import { Chart } from "chart.js";
import { FederatedSiteService } from "../../../services/federatedSiteService.service";
import { Worker } from "../../../models/worker.model";
import { Coordinator } from "../../../models/coordinator.model";
import { Statistics } from "../../../models/statistics.model";
import { constants } from "../../../constants";
import 'chartjs-adapter-moment';
import { Subject } from "rxjs";
import { Utils } from "../../../utils";

@Component({
	selector: 'app-connection',
	templateUrl: './connection.component.html',
	styleUrls: ['./connection.component.scss']
})
export class ConnectionComponent implements OnInit {

	public workerId: number;

	public worker: Worker;
	public coordinator: Coordinator;
	public statistics: Statistics;

	private stopPollingStatistics = new Subject<any>();

	constructor(private fedSiteService: FederatedSiteService) { }

	ngOnInit(): void {
		this.statistics = new Statistics();

		const id = `traffic-${constants.prefixes.coordinator + this.coordinator.id + constants.prefixes.worker + this.worker.id}`;

		const trafficMetricEle: any = document.getElementById(id);

		let trafficChart = new Chart(trafficMetricEle.getContext('2d'), {
			type: 'line',
			data: {
				datasets: [{
					data: this.statistics.utilization.map(s => {
						return { x: s.timestamp, y: s.memoryUsage }
					}),
					borderColor: constants.chartColors.green
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
						text: 'I/O Bytes'
					}
				},
				scales: {
					x: {
						grid: {
							display: false
						},
						type: 'timeseries',
						ticks: {
							display: false,
						}
					},
					y: {
						beginAtZero: true,
					}
				}
			},
		});

		this.fedSiteService.getStatisticsPolling(this.workerId, this.stopPollingStatistics).subscribe(stats => {
			this.statistics = stats;

			trafficChart.data.datasets.forEach((dataset) => {
				dataset.data = [];
				this.statistics.traffic.map(s => dataset.data.push({ x: s.timestamp, y: s.byteAmount }));
				dataset.data.sort(Utils.sortTimestamp);
			});

			trafficChart.update();
		});
	}

	ngOnDestroy() {
		this.stopPollingStatistics.next(null);
	}
}
