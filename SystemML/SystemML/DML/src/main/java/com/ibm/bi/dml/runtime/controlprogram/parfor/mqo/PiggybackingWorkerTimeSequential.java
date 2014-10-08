/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.util.LinkedList;

import com.ibm.bi.dml.lops.runtime.RunMRJobs;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.io.Pair;
import com.ibm.bi.dml.utils.Statistics;

public class PiggybackingWorkerTimeSequential extends PiggybackingWorker
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

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
