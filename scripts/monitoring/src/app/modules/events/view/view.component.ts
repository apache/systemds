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
import { constants } from "../../../constants";
import 'chartjs-adapter-moment';
import { Subject } from "rxjs";

@Component({
	selector: 'app-view-worker-events',
	templateUrl: './view.component.html',
	styleUrls: ['./view.component.scss']
})
export class ViewWorkerEventsComponent {

	public statistics: Statistics;

	@ViewChild(MatPaginator) paginator: MatPaginator;
	@ViewChild(MatSort) sort: MatSort;

	private eventTimelineChart: Chart;

	private stopPollingStatistics = new Subject<any>();

	constructor(
		private fedSiteService: FederatedSiteService,
		private router: ActivatedRoute) {
		Chart.register(...registerables);
	}

	ngOnInit(): void {
		const id = Number(this.router.snapshot.paramMap.get('id'));

		this.statistics = new Statistics();

		const eventCanvasEle: any = document.getElementById('event-timeline');

		this.fedSiteService.getStatisticsPolling(id, this.stopPollingStatistics).subscribe(stats => {
			this.statistics = stats;

			const timeframe = this.getTimeframe();

			if (!this.eventTimelineChart) {
				this.eventTimelineChart = new Chart(eventCanvasEle.getContext('2d'), {
					type: 'bar',
					data: {
						labels: [],
						datasets: []
					},
					options: {
						indexAxis: 'y',
						responsive: true,
						plugins: {
							legend: {
								position: 'top',
							},
							title: {
								display: true,
								text: 'Event timeline of worker with respect to coordinators'
							}
						},
						scales: {
							x: {
								min: this.getLastSeconds(timeframe[1], 3),
								max: timeframe[1],
								ticks: {
									callback: function(value, index, ticks) {
										return new Date(value).toLocaleTimeString();
									}
								}
							}
						}
					}
				})
			}

			this.updateEventTimeline();
		});
	}

	private getLastSeconds(time: number, seconds: number): number {
		const benchmark = new Date(time);

		const back = new Date(time);
		back.setSeconds(benchmark.getSeconds() - seconds)

		return back.getTime();
	}

	private getTimeframe() {
		const coordinatorNames = this.getCoordinatorNames();
		let minTime = 0;
		let maxTime = 0;

		coordinatorNames.forEach(c => {
			const eventsData = this.getEventsData(c);

			for (const entry in eventsData) {
				let startTime = new Date(eventsData[entry]['startTime']).getTime();
				let endTime = new Date(eventsData[entry]['endTime']).getTime();

				if (startTime < minTime) {
					minTime = startTime;
				}

				if (endTime > maxTime) {
					maxTime = endTime;
				}
			}
		})

		return [minTime, maxTime];
	}

	private getCoordinatorNames() {
		let names: string[] = [];

		this.statistics.events.forEach(e => {
			if (!names.find(n => n === e.coordinatorName)) {
				names.push(e.coordinatorName);
			}
		})

		return names;
	}

	private updateEventTimeline() {
		const coordinatorNames = this.getCoordinatorNames();
		coordinatorNames.forEach(c => {
			const eventsData = this.getEventsData(c);

			this.eventTimelineChart.data.datasets = [];
			this.eventTimelineChart.data.labels = coordinatorNames;

			for (const entry in eventsData) {
				let startTime = new Date(eventsData[entry]['startTime']).getTime();
				let endTime = new Date(eventsData[entry]['endTime']).getTime();

				this.eventTimelineChart.data.datasets.push({
					label: entry,
					backgroundColor: constants.chartColors.green,
					// @ts-ignore
					data: [[startTime, endTime]]
				})
			}

			this.eventTimelineChart.update('none');
		})
	}

	private getEventsData(coordinatorName: string) {

		let result: any = {};

		for (let i = 0; i < this.statistics.events.length; i++) {
			const event = this.statistics.events[i];

			if (event.coordinatorName === coordinatorName) {
				for (let j = 0; j < event.stages.length; j++) {
					const stage = event.stages[j];

					if (result[stage.operation] && stage.startTime < result[stage.operation]['startTime']) {
						continue;
					}

					result[stage.operation] = stage;
				}
			}
		}

		return result;
	}

	ngOnDestroy() {
		this.stopPollingStatistics.next(null);
	}

}
