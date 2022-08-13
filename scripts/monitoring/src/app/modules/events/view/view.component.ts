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
import { Chart, LegendItem, registerables } from "chart.js";
import { constants } from "../../../constants";
import 'chartjs-adapter-moment';
import { Subject } from "rxjs";
import { EventStage } from "../../../models/eventStage.model";

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
			const minVal = this.getLastSeconds(timeframe[1], 3);

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
								onClick: () => null,
								onHover: () => null,
								onLeave: () => null,
								labels: {
									generateLabels(chart: Chart): LegendItem[] {
										let legendItemsTmp: LegendItem[] = [];

										for (const dataset of chart.data.datasets) {
											const label = dataset.label!
											if (!legendItemsTmp.find(i => i.text === label)) {
												let li: LegendItem = {
													text: label,
													//@ts-ignore
													fillStyle: dataset.backgroundColor,
													//@ts-ignore
													strokeStyle: dataset.borderColor,
												}
												legendItemsTmp.push(li);
											}
										}

										return legendItemsTmp;
									}
								}
							},
							title: {
								display: true,
								text: 'Event timeline of worker with respect to coordinators'
							}
						},
						scales: {
							x: {
								min: 0,
								ticks: {
									callback: function(value, index, ticks) {
										// @ts-ignore
										return new Date(minVal + value).toLocaleTimeString();
									}
								},
								stacked: true
							},
							y: {
								stacked: true
							}
						},
					},
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
			const coordinatorEvents = this.statistics.events.filter(e => e.coordinatorName === c);

			for (const event of coordinatorEvents) {
				for (const stage of event.stages) {
					let startTime = new Date(stage.startTime).getTime();
					let endTime = new Date(stage.endTime).getTime();

					if (startTime < minTime) {
						minTime = startTime;
					}

					if (endTime > maxTime) {
						maxTime = endTime;
					}
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

	private getColor(operation: string) {

		let hash = 0
		for (let x = 0; x < operation.length; x++) {
			let ch = operation.charCodeAt(x);
			hash = ((hash <<5) - hash) + ch;
			hash = hash & hash;
		}

		let r = (hash & 0xFF0000) >> 16;
		let g = (hash & 0x00FF00) >> 8;
		let b = hash & 0x0000FF;

		return `rgb(${r}, ${g}, ${b}, 0.8)`;
	}

	private updateEventTimeline() {
		const coordinatorNames = this.getCoordinatorNames();
		coordinatorNames.forEach(c => {

			this.eventTimelineChart.data.datasets = [];
			this.eventTimelineChart.data.labels = [coordinatorNames];

			let coordinatorEvents = this.statistics.events.filter(e => e.coordinatorName === c);

			let stageStack: EventStage[] = [];

			for (let eventIndex = 0; eventIndex < coordinatorEvents.length; eventIndex++) {
				const event = coordinatorEvents[eventIndex];

				if (event.stages.length > 1) {
					for (let stageIndex = 1; stageIndex < event.stages.length; stageIndex++) {
						let currentStage = stageStack.pop();
						if (!currentStage) {
							currentStage = event.stages[stageIndex - 1];
						}
						let nextStage = event.stages[stageIndex];
						stageStack.push(nextStage);

						this.eventTimelineChart.data.datasets.push({
							type: 'bar',
							label: currentStage.operation,
							backgroundColor: this.getColor(currentStage.operation),
							data: [new Date(currentStage.endTime).getTime() - new Date(currentStage.startTime).getTime()]
						});

						this.placeIntermediateBars(currentStage, nextStage);
					}
				} else {
					stageStack.push(event.stages[0]);
				}

				const lastStage = stageStack.pop()!;

				this.eventTimelineChart.data.datasets.push({
					type: 'bar',
					label: lastStage.operation,
					borderColor: constants.chartColors.red,
					borderWidth: {
						top: 0,
						bottom: 0,
						left: 0,
						right: 4
					},
					backgroundColor: this.getColor(lastStage.operation),
					data: [new Date(lastStage.endTime).getTime() - new Date(lastStage.startTime).getTime()]
				});
			}

			this.eventTimelineChart.update('none');
		})
	}

	private placeIntermediateBars(first: EventStage, second: EventStage) {
		let firstEnd = new Date(first.endTime).getTime();
		let secondStart = new Date(second.startTime).getTime();

		let diff = secondStart - firstEnd;

		if (diff > 0) {
			this.eventTimelineChart.data.datasets.push({
				type: 'bar',
				label: 'Idle',
				backgroundColor: constants.chartColors.white,
				data: [diff]
			});
		} else if (diff < 0) {
			this.eventTimelineChart.data.datasets.push({
				type: 'bar',
				label: 'Overlap',
				backgroundColor: constants.chartColors.grey,
				data: [Math.abs(diff)]
			});
		}
	}

	ngOnDestroy() {
		this.stopPollingStatistics.next(null);
	}

}
