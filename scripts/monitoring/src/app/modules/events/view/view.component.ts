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
import { FederatedSiteService } from 'src/app/services/federatedSiteService.service';
import { Statistics } from "../../../models/statistics.model";
import { Chart, registerables } from "chart.js";

@Component({
	selector: 'app-view-worker-events',
	templateUrl: './view.component.html',
	styleUrls: ['./view.component.scss']
})
export class ViewWorkerEventsComponent {

	public optionsEvents: any;

	public updateOptionsEvents: any;
	public statistics: Statistics;
	@ViewChild(MatPaginator) paginator: MatPaginator;
	@ViewChild(MatSort) sort: MatSort;
	private dataEvents!: any[];
	private timer: any;

	constructor(
		private fedSiteService: FederatedSiteService,
		private router: ActivatedRoute) {
		Chart.register(...registerables);
	}

	ngOnInit(): void {
		const id = Number(this.router.snapshot.paramMap.get('id'));

		this.fedSiteService.getStatistics(id).subscribe(stats => {
			this.statistics = stats;

			this.updateMetrics();
		});

		const DATA_COUNT = 7;
		const NUMBER_CFG = {count: DATA_COUNT, min: -100, max: 100};

		const labels = [
			'coordinator 1',
		];
		const data = {
			labels: labels,
			datasets: [
				{
					label: 'Dataset 1',
					data: [[3, 7]],
					backgroundColor: 'rgb(255, 99, 132)',
				},
				{
					label: 'Dataset 2',
					data: labels.map(() => {
						return [Math.random() * (100 + 100) - 100, Math.random() * (100 + 100) - 100];
					}),
					backgroundColor: 'rgb(54, 162, 235)',
				},

				{
					label: 'Dataset 3',
					data: labels.map(() => {
						return [Math.random() * (100 + 100) - 100, Math.random() * (100 + 100) - 100];
					}),
					backgroundColor: 'rgb(3,86,11)',
				},
			]
		};

		const eventCanvasEle: any = document.getElementById('event-timeline');

		new Chart(eventCanvasEle.getContext('2d'), {
			type: 'bar',
			data: data,
			options: {
				indexAxis: 'y',
				responsive: true,
				plugins: {
					legend: {
						position: 'top',
					},
					title: {
						display: true,
						text: 'Chart.js Floating Bar Chart'
					}
				},
				scales: {
					x: {
						stacked: true
					},
					y: {
						stacked: true
					}
				}
			}
		});

		// this.timer = setInterval(() => {
		// 	this.fedSiteService.getStatistics(this.model.id).subscribe(stats => {
		// 		this.statistics = stats;
		//
		// 		this.updateMetrics();
		// 	})
		// }, 3000);

	}

	ngOnDestroy() {
		clearInterval(this.timer);
	}

	private getDatasets() {
		this.statistics.events.map(e => {

		})
	}

	private updateMetrics(): void {
		this.dataEvents = this.statistics.utilization.map(s => {
			return {
				name: s.timestamp,
				value: [
					s.timestamp,
					s.cpuUsage
				]
			}
		});

		this.updateOptionsEvents = {
			series: [{
				data: this.dataEvents
			}]
		};
	}

}
