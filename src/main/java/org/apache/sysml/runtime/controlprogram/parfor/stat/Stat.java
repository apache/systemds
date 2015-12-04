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

package org.apache.sysml.runtime.controlprogram.parfor.stat;

/**
 * Collection of all ParFor statistic types.
 * 
 *
 */
public enum Stat
{
	//parfor parser statistics
	PARSE_T,
	//parfor optimizer statistics
	OPT_T,
	OPT_OPTIMIZER,
	OPT_NUMTPLANS,
	OPT_NUMEPLANS,
	//parfor program block statistics
	PARFOR_NUMTHREADS,
	PARFOR_TASKSIZE,
	PARFOR_TASKPARTITIONER,
	PARFOR_DATAPARTITIONER,
	PARFOR_EXECMODE,	
	PARFOR_NUMTASKS,
	PARFOR_NUMITERS,
	PARFOR_JITCOMPILE,
	PARFOR_JVMGC_COUNT,
	PARFOR_JVMGC_TIME,
	PARFOR_INIT_DATA_T,
	PARFOR_INIT_PARWRK_T,
	PARFOR_INIT_TASKS_T,
	PARFOR_WAIT_EXEC_T,
	PARFOR_WAIT_RESULTS_T,
	
	//parallel worker statistics
	PARWRK_NUMTASKS,
	PARWRK_NUMITERS,
	PARWRK_TASKSIZE,
	PARWRK_ITER_T,
	PARWRK_TASK_T,
	PARWRK_EXEC_T;
	

}
