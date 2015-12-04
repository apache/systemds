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

package org.apache.sysml.runtime.controlprogram.parfor.mqo;

import java.util.LinkedList;

import org.apache.sysml.lops.runtime.RunMRJobs;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.matrix.JobReturn;
import org.apache.sysml.runtime.matrix.data.Pair;
import org.apache.sysml.utils.Statistics;

public class PiggybackingWorkerTimeSequential extends PiggybackingWorker
{

	//internal configuration parameters
	private static long DEFAULT_MERGE_INTERVAL = 1000;
	private static boolean SUBSTRACT_EXEC_TIME = true;
	
	private long _time;
	
	public PiggybackingWorkerTimeSequential()
	{
		this(DEFAULT_MERGE_INTERVAL);
	}
	
	public PiggybackingWorkerTimeSequential( long timeInterval )
	{
		_time = timeInterval;
	}

	@Override
	public void run() 
	{
		long lastTime = System.currentTimeMillis();
		
		while( !_stop )
		{
			try
			{
				// wait until next submission
				if( SUBSTRACT_EXEC_TIME ) {
					long currentTime = System.currentTimeMillis();
					if( currentTime-lastTime < _time  )
						Thread.sleep( _time-(currentTime-lastTime) );
					lastTime = currentTime;
				}
				else
					Thread.sleep(_time);
				
				
				// pick job type with largest number of jobs
				LinkedList<Pair<Long,MRJobInstruction>> workingSet = RuntimePiggybacking.getMaxWorkingSet();
				if( workingSet == null )
					continue; //empty pool
				
				// merge jobs (if possible)
				LinkedList<MergedMRJobInstruction> mergedWorkingSet = mergeMRJobInstructions(workingSet);
				
				// submit all resulting jobs (currently sequential submission)
				for( MergedMRJobInstruction minst : mergedWorkingSet )
				{
					JobReturn mret = RunMRJobs.submitJob(minst.inst);
					Statistics.incrementNoOfExecutedMRJobs();
					
					// error handling
					if( !mret.successful )
						LOG.error("Failed to run merged mr-job instruction:\n"+minst.inst.toString()); 
					
					// split job return
					LinkedList<JobReturn> ret = new LinkedList<JobReturn>();
					for( Long id : minst.ids ){
						ret.add( minst.constructJobReturn(id, mret) );
						Statistics.decrementNoOfExecutedMRJobs();
					}
					// make job returns available and notify waiting clients
					putJobResults(minst.ids, ret);
				}
			}
			catch(Exception ex)
			{
				throw new RuntimeException(ex);
			}
		}
	}
}
