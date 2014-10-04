/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.util.HashMap;
import java.util.LinkedList;


import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.io.Pair;


/**
 * 
 * 
 */
public class RuntimePiggybacking 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum PiggybackingType {
		TIME_BASED_SEQUENTIAL,
		UTIL_TIME_BASED_PARALLEL,
		UTIL_DECAY_BASED_PARALLEL,
	}
	
	private static boolean _active = false;
	private static IDSequence _idSeq = null;
	private static PiggybackingWorker _worker = null;
	
	//mr instruction pool
	private static HashMap<JobType, LinkedList<Long>> _pool = null;
	private static HashMap<Long, MRJobInstruction> _jobs = null;
	
	
	//static initialization of piggybacking server
	static
	{
		//initialize mr-job instruction pool
		_pool = new HashMap<JobType, LinkedList<Long>>();	
		_jobs = new HashMap<Long, MRJobInstruction>();
		
		//init id sequence
		_idSeq = new IDSequence();
	}

	
	/////////////////////////////////////////////////
	// public interface to runtime piggybacking
	///////
	
	/**
	 * 
	 * @return
	 */
	public static boolean isActive()
	{
		return _active;
	}
	
	/**
	 * @throws DMLRuntimeException 
	 * 
	 */
	public static void start( PiggybackingType type, int par ) 
		throws DMLRuntimeException
	{
		//activate piggybacking server
		_active = true;
		
		//init job merge/submission worker 
		switch( type )
		{
			case TIME_BASED_SEQUENTIAL:
				_worker = new PiggybackingWorkerTimeSequential();
				break;
			case UTIL_TIME_BASED_PARALLEL:
				_worker = new PiggybackingWorkerUtilTimeParallel(par);
				break;
			case UTIL_DECAY_BASED_PARALLEL:
				_worker = new PiggybackingWorkerUtilDecayParallel(par);
				break;
			default:
				throw new DMLRuntimeException("Unsupported runtime piggybacking type: "+type);
		}
		
		//start worker
		_worker.start();
	}
	
	/**
	 * 
	 * @throws DMLRuntimeException
	 */
	public static void stop() 
		throws DMLRuntimeException 
	{
		try
		{
			//deactivate piggybacking server
			_active = false;
			
			//cleanup merge/submission worker
			_worker.setStopped();
			_worker.join();
			_worker = null;
		}
		catch(InterruptedException ex)
		{
			throw new DMLRuntimeException("Failed to stop runtime piggybacking server.", ex);
		}
	}
	
	/**
	 * 
	 * @param inst
	 * @param ec
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static JobReturn submitJob(MRJobInstruction inst) 
		throws DMLRuntimeException 
	{
		JobReturn ret = null;
		
		try
		{
			//step 1: obtain job id
			long id = _idSeq.getNextID();
			
			//step 2: append mr job to global pool
			synchronized( _pool )
			{
				//maintain job-type partitioned instruction pool 
				if( !_pool.containsKey(inst.getJobType()) )
					_pool.put(inst.getJobType(), new LinkedList<Long>());
				_pool.get(inst.getJobType()).add( id );	
				
				//add actual mr job instruction
				_jobs.put(id, inst);
			}
			
			//step 3: wait for finished job
			ret = _worker.getJobResult( id );
		}
		catch(InterruptedException ex)
		{
			throw new DMLRuntimeException("Failed to submit MR job to runtime piggybacking server.", ex);
		}
		
		return ret;
	}		
	
	/**
	 * Gets a working set of MR job instructions out of the global pool.
	 * All returned instructions are guaranteed to have the same job type
	 * but are not necessarily mergable. This method returns the largest 
	 * instruction set currently in the pool. 
	 * 
	 * @return
	 */
	protected static LinkedList<Pair<Long,MRJobInstruction>> getMaxWorkingSet()
	{
		LinkedList<Pair<Long,MRJobInstruction>> ret = null;	
		
		synchronized( _pool )
		{
			//determine job type with max number of 
			JobType currType = null;
			int currLength = 0;
			for( JobType jt : _pool.keySet() ){
				LinkedList<Long> tmp = _pool.get(jt);
				if( tmp!=null && currLength<tmp.size() ){
					currLength = tmp.size();
					currType = jt;
				}
			}
	
			//indicate empty pool if necessary
			if( currType==null )
				return null;
				
			//create working set and remove from pool
			ret = new LinkedList<Pair<Long,MRJobInstruction>>();
			LinkedList<Long> tmp = _pool.remove(currType);
			for( Long id : tmp )
				ret.add( new Pair<Long, MRJobInstruction>(id,_jobs.get(id)) );
		}
		
		return ret;
	}
}
