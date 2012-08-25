package com.ibm.bi.dml.runtime.controlprogram.parfor.stat;

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
	PARWRK_EXEC_T
}
