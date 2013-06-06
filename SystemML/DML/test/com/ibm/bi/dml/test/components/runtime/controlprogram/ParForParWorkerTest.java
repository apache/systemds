package com.ibm.bi.dml.test.components.runtime.controlprogram;


import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;

import junit.framework.Assert;

import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.SymbolTable;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ParForBody;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ParWorker;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * TestCases for local task queue.
 * 
 */
public class ParForParWorkerTest extends ParWorker
{
	private static final int _N = 101;
	
	
	public void testExecutionSet() 
		throws InterruptedException, DMLRuntimeException, DMLUnsupportedOperationException 
	{ 
		Collection<Task> tasks = createTasks( TaskType.SET );
		setParWorkerAttributes();
		
		for( Task ltask : tasks )
			executeTask(ltask);
		
		if( _numIters!=_N || _numTasks!=_N )
			Assert.fail("Wrong number of executed tasks or iterations.");
	}
	
	
	public void testExecutionRange() 
		throws InterruptedException, DMLRuntimeException, DMLUnsupportedOperationException 
	{ 
		Collection<Task> tasks = createTasks( TaskType.RANGE );
		setParWorkerAttributes();
		
		for( Task ltask : tasks )
			executeTask(ltask);
		
		if( _numIters!=_N || _numTasks!=1 )
			Assert.fail("Wrong number of executed tasks or iterations.");
	}
	
	
	private Collection<Task> createTasks(TaskType type) 
		throws DMLRuntimeException
	{
		//create N tasks
		Collection<Task> tasks = new LinkedList<Task>();
		
		if( type == TaskType.SET )
		{
			for( int i=1; i<=_N; i++ )
			{
				Task ltask = new Task(TaskType.SET);
				ltask.addIteration(new IntObject("i",i));
				tasks.add(ltask);
	 		}
		}
		else if(type == TaskType.RANGE)
		{
			Task ltask = new Task(TaskType.RANGE);
			ltask.addIteration(new IntObject("i",1));
			ltask.addIteration(new IntObject("i",_N));
			ltask.addIteration(new IntObject("i",1));
			tasks.add(ltask);
		}
		else
			throw new DMLRuntimeException("Undefined task type.");
		
		return tasks;
	}
	
	
	private void setParWorkerAttributes() throws DMLRuntimeException
	{
		Program prog = new Program();
		ArrayList<ProgramBlock> pbs = new ArrayList<ProgramBlock>();
		pbs.add(new ProgramBlock(prog));
		ExecutionContext ec = new ExecutionContext();
		SymbolTable symb = prog.createSymbolTable();
		ec.setSymbolTable(symb);
		ParForBody body = new ParForBody(pbs,new ArrayList<String>(),ec);
		
		_workerID = -1;
		_numTasks = 0;
		_numIters = 0;
		//_variables = body.getVariables();
		_childBlocks = body.getChildBlocks();
		_ec = body.getEc();
	}
}
