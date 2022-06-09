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

@Component({
  selector: 'app-view-worker',
  templateUrl: './view.component.html',
  styleUrls: ['./view.component.scss']
})
export class ViewWorkerComponent {

  public optionsCPU: any;
  public optionsMemory: any;

  public updateOptionsCPU: any;
  public updateOptionsMemory: any;

  private dataCPU!: any[];
  private dataMemory!: any[];

  private timer: any;

  public model: Worker;

  public displayedColumns: string[] = ['type', 'executionTime'];
  public resultsLength = 0;
  public isLoadingResults = true;

  @ViewChild(MatPaginator) paginator: MatPaginator;
  @ViewChild(MatSort) sort: MatSort;

  constructor(
    private fedSiteService: FederatedSiteService,
    private router: ActivatedRoute) { }

  ngOnInit(): void {
    const id = Number(this.router.snapshot.paramMap.get('id'));
    this.fedSiteService.getWorker(id).subscribe(worker => {
      this.model = worker;

      this.isLoadingResults = false;
      this.resultsLength = this.model.stats!.length;

      this.updateMetrics();
    });

    this.dataCPU = [];
    this.dataMemory = [];

    // initialize chart options:
    this.optionsCPU = {
      title: {
        text: 'CPU (%)'
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
        areaStyle: { },
        data: this.dataCPU
      }]
    };

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
        areaStyle: { },
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
    this.dataCPU = this.model.stats.map(s => {
      return {
        name: s.timestamp,
        value: [
          s.timestamp,
          s.cpuUsage
        ]
      }
    });

    this.dataMemory = this.model.stats.map(s => {
      return {
        name: s.timestamp,
        value: [
          s.timestamp,
          s.memoryUsage
        ]
      }
    })

    this.updateOptionsCPU = {
      series: [{
        data: this.dataCPU
      }]
    };

    // update series data:
    this.updateOptionsMemory = {
      series: [{
        data: this.dataMemory
      }]
    };
  }

}
