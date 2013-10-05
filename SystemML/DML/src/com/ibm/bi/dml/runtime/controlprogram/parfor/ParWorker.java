/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor;

import java.util.ArrayList;
import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * Super class for master/worker pattern implementations. Central place to
 * execute set or range tasks.
 * 
 */
public abstract class ParWorker
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected static final Log LOG = LogFactory.getLog(ParWorker.class.getName());
	
	protected long                      _workerID    = -1;
	
	protected ArrayList<ProgramBlock>   _childBlocks = null;
	protected ExecutionContext          _ec          = null;
	protected ArrayList<String>         _resultVars  = null;
	
	protected int                       _numTasks    = -1;
	protected int                       _numIters    = -1;
	
	public ParWorker()
	{
		//implicit constructor (required if parameters not known on object creation, 
		//e.g., RemoteParWorkerMapper)
	}
	
	public ParWorker( long ID, ParForBody body )
	{
		_workerID    = ID;
		
		if( body != null )
		{
			_childBlocks = body.getChildBlocks();
			_ec          = body.getEc();
			_resultVars  = body.getResultVarNames();
		}
		
		_numTasks    = 0;
		_numIters    = 0;
	}
	
	/**
	 * 
	 * @return
	 */
	public LocalVariableMap getVariables()
	{
		return _ec.getVariables();
	}
	
	/**
	 * Returns a summary statistic of executed tasks and hence should only be called 
	 * after execution.
	 * 
	 * @return
	 */
	public int getExecutedTasks()
	{
		return _numTasks;
	}
	
	/**
	 * Returns a summary statistic of executed iterations and hence should only be called 
	 * after execution.
	 * 
	 * @return
	 */
	public int getExecutedIterations()
	{
		return _numIters;
	}

	/**
	 * 
	 * @param task
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void executeTask( Task task ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		LOG.trace("EXECUTE PARFOR_WORKER ID="+_workerID+" for task "+task.toCompactString());
		
		switch( task.getType() )
		{
			case SET:
				executeSetTask( task );
				break;
			case RANGE:
				executeRangeTask( task );
				break;		
			//default: //MB: due to enum types this can never happen
			//	throw new DMLRuntimeException("Undefined task type: '"+task.getType()+"'.");
		}
	}
		
		
	/**
	 * 
	 * @param task
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private void executeSetTask( Task task ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//monitoring start
		Timing time1, time2;		
		if( ParForProgramBlock.MONITOR )
		{
			time1 = new Timing(); time1.start();
			time2 = new Timing(); time2.start();
		}
		
		//core execution

		//foreach iteration in task, execute iteration body
		for( IntObject indexVal : task.getIterations() )
		{
			//System.out.println(" EXECUTE ITERATION: "+indexVal.getName()+"="+indexVal.getIntValue());
			
			//set index values
			_ec.setVariable(indexVal.getName(), indexVal);
			
			// for each program block
			for (ProgramBlock pb : _childBlocks)
				pb.execute(_ec);
					
			_numIters++;
			
			if(ParForProgramBlock.MONITOR)
				StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_ITER_T, time1.stop());
		}

		_numTasks++;
		
		//monitoring end
		if(ParForProgramBlock.MONITOR)
		{
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_TASKSIZE, task.size());
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_TASK_T, time2.stop());
		}
	}
	
	/**
	 * 
	 * @param task
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private void executeRangeTask( Task task ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//monitoring start
		Timing time1, time2;		
		if( ParForProgramBlock.MONITOR )
		{
			time1 = new Timing(); time1.start();
			time2 = new Timing(); time2.start();
		}
		
		//core execution
		LinkedList<IntObject> tmp = task.getIterations();
		String lVarName = tmp.get(0).getName();
		int lFrom       = tmp.get(0).getIntValue();
		int lTo         = tmp.get(1).getIntValue();
		int lIncr       = tmp.get(2).getIntValue();
		
		for( int i=lFrom; i<=lTo; i+=lIncr )
		{
			//set index values
			_ec.setVariable(lVarName, new IntObject(lVarName,i));
			
			// for each program block
			for (ProgramBlock pb : _childBlocks)
				pb.execute(_ec);
					
			_numIters++;
			
			if(ParForProgramBlock.MONITOR)
				StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_ITER_T, time1.stop());	
		}

		_numTasks++;
		
		//monitoring end
		if(ParForProgramBlock.MONITOR)
		{
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_TASKSIZE, task.size());
			StatisticMonitor.putPWStat(_workerID, Stat.PARWRK_TASK_T, time2.stop());
		}
	}
		
}

	
