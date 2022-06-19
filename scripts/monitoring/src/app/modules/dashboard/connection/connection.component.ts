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

import { Component, OnInit, OnDestroy } from '@angular/core';

@Component({
	selector: 'app-connection',
	templateUrl: './connection.component.html',
	styleUrls: ['./connection.component.scss']
})
export class ConnectionComponent implements OnInit, OnDestroy {
	public options: any;
	public updateOptions: any;

	private oneDay = 24 * 3600 * 1000;
	private now!: Date;
	private value!: number;
	private data!: any[];
	private timer: any;

	constructor() {
	}

	ngOnInit(): void {
		// generate some random testing data:
		this.data = [];
		this.now = new Date();
		this.value = Math.random() * 10;

		for (let i = 0; i < 20; i++) {
			this.data.push(this.randomData());
		}

		// initialize chart options:
		this.options = {
			title: {
				text: 'I/O (Byte amount)'
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
				data: this.data
			}]
		};

		// Mock dynamic data:
		this.timer = setInterval(() => {

			this.data.shift();
			this.data.push(this.randomData());

			// update series data:
			this.updateOptions = {
				series: [{
					data: this.data
				}]
			};
		}, 1000);
	}

	ngOnDestroy() {
		clearInterval(this.timer);
	}

	randomData() {
		this.now = new Date(this.now.getTime() + this.oneDay);
		this.value = Math.random() * 10;
		return {
			name: this.now.toString(),
			value: [
				[this.now.getFullYear(), this.now.getMonth() + 1, this.now.getDate()].join('/'),
				Math.round(this.value)
			]
		};
	}
}
