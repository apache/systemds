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
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.io.Pair;


/**
 * 
 * TODO extended runtime support: variable patching, call-backs instead of time based polling
 */
public class RuntimePiggybacking 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum PiggybackingType {
		TIME_BASED_SEQUENTIAL,
		TIME_BASED_PARALLEL,
		UTIL_BASED_SEQUENTIAL,
		UTIL_BASED_PARALLEL,
	}
	
	//internal configuration parameters
	private static long DEFAULT_MERGE_INTERVAL = 1000;
	
	private static boolean _active = false;
	private static IDSequence _idSeq = null;
	private static PiggybackingWorker _worker = null;
	
	//mr instruction pool
	private static HashMap<JobType, LinkedList<Long>> _pool = null;
	private static HashMap<Long, MRJobInstruction> _jobs = null;
	private static HashMap<Long, JobReturn> _results = null;
	
	
	//static initialization of piggybacking server
	static
	{
		//initialize mr-job instruction pool
		_pool = new HashMap<JobType, LinkedList<Long>>();	
		_jobs = new HashMap<Long, MRJobInstruction>();
		_results = new HashMap<Long, JobReturn>();
		
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
	 * 
	 */
	public static void start( PiggybackingType type )
	{
		//activate piggybacking server
		_active = true;
		
		//init job merge/submission worker 
		_worker = new PiggybackingWorkerSequential(DEFAULT_MERGE_INTERVAL);
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
	public static JobReturn submitJob(MRJobInstruction inst, ExecutionContext ec) 
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
			//TODO rework communication
			while( ret == null ){
				//_worker.wait(); //see notify in worker
				Thread.sleep(500);
				synchronized( _results ){
					ret = _results.remove(id);
				}
			}
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
	
	/**
	 * 
	 * @param ids
	 * @param results
	 */
	protected static void putWorkingSetJobResults( LinkedList<Long> ids, LinkedList<JobReturn> results )
	{
		synchronized( _results )
		{
			for( int i=0; i<ids.size(); i++ )
				_results.put(ids.get(i), results.get(i));
		}
	}
}
