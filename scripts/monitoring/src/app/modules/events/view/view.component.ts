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

@Component({
	selector: 'app-view-worker-events',
	templateUrl: './view.component.html',
	styleUrls: ['./view.component.scss']
})
export class ViewWorkerEventsComponent {

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

		this.statistics = new Statistics();

		const eventCanvasEle: any = document.getElementById('event-timeline');

		const eventTimelineChart = new Chart(eventCanvasEle.getContext('2d'), {
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
						type: 'time',
						time: {
							unit: 'millisecond'
						}
					}
				}
			}
		})

		this.timer = setInterval(() => {
			this.fedSiteService.getStatistics(id).subscribe(stats => {
				this.statistics = stats;

				this.updateEventTimeline(eventTimelineChart);
			})
		}, 3000);

	}

	ngOnDestroy() {
		clearInterval(this.timer);
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

	private updateEventTimeline(chart: Chart) {
		const coordinatorNames = this.getCoordinatorNames();
		coordinatorNames.forEach(c => {
			const eventsData = this.getEventsData(c);

			chart.data.datasets = [];
			chart.data.labels = coordinatorNames;

			for (const entry in eventsData) {
				chart.data.datasets.push({
					label: entry,
					backgroundColor: constants.chartColors.green,
					// @ts-ignore
					data: [[new Date(eventsData[entry]['startTime']).getTime(), new Date(eventsData[entry]['endTime']).getTime()]]
				})
			}

			chart.update('none');
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

}
