/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.util.HashMap;
import java.util.LinkedList;


import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.data.Pair;


/**
 * 
 * 
 */
public class RuntimePiggybacking 
{
	
	public enum PiggybackingType {
		TIME_BASED_SEQUENTIAL,
		UTIL_TIME_BASED_PARALLEL,
		UTIL_DECAY_BASED_PARALLEL,
	}
	
	private static PiggybackingType DEFAULT_WORKER_TYPE = PiggybackingType.UTIL_DECAY_BASED_PARALLEL;
	
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
	 * 
	 * @param type
	 * @param par
	 * @throws DMLRuntimeException
	 */
	public static void start( int par ) 
		throws DMLRuntimeException
	{
		start( DEFAULT_WORKER_TYPE, par );
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
			
			if( !ret.successful )
				throw new DMLRuntimeException("Failed to run MR job via runtime piggybacking - job unsuccessful:\n"+inst.toString());
		}
		catch(InterruptedException ex)
		{
			throw new DMLRuntimeException("Failed to submit MR job to runtime piggybacking server.", ex);
		}
		
		return ret;
	}		
	
	/**
	 * 
	 * @param type
	 * @return
	 */
	public static boolean isSupportedJobType( JobType type )
	{
		// reblock and datagen apply as well but this would limit the recompilation
		// potential of job-specific recompilation hooks due all-or-nothing semantics
		
		return (   type == JobType.GMR
				|| type == JobType.CM_COV
				|| type == JobType.GROUPED_AGG
				|| type == JobType.REBLOCK );
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
	 * @return
	 */
	public static boolean isEmptyJobPool()
	{
		return _pool.isEmpty();
	}
}
