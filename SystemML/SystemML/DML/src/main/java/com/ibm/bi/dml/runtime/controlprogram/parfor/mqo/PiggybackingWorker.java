/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.io.Pair;

public abstract class PiggybackingWorker extends Thread
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	protected static final Log LOG = LogFactory.getLog(PiggybackingWorker.class.getName());
	
	protected boolean _stop;
	
	protected PiggybackingWorker()
	{
		_stop = false;
	}

	public void setStopped()
	{
		_stop = true;
	}
	
	/**
	 * 
	 * @param workingSet
	 * @return
	 * @throws IllegalAccessException 
	 * @throws IllegalArgumentException 
	 */
	protected LinkedList<MergedMRJobInstruction> mergeMRJobInstructions( LinkedList<Pair<Long,MRJobInstruction>> workingSet ) 
		throws IllegalArgumentException, IllegalAccessException
	{
		LinkedList<MergedMRJobInstruction> ret = new LinkedList<MergedMRJobInstruction>();
		
		//NOTE currently all merged into one (might be invalid due to memory constraints)
		MergedMRJobInstruction minst = new MergedMRJobInstruction();
		for( Pair<Long,MRJobInstruction> inst : workingSet )
		{
			long instID = inst.getKey();
			MRJobInstruction instVal = inst.getValue();
			int numOutputs = instVal.getOutputs().length;
			
			//append to current merged instruction
			if( minst.inst==null )
			{
				//deep copy first instruction
				minst.inst = new MRJobInstruction( instVal );	
				minst.addInstructionMetaData( instID, 0, numOutputs );
			}
			else
			{	
				//merge other instructions
				if( minst.inst.isMergableMRJobInstruction( instVal ) )
				{
					//add instruction to open merged instruction
					int offOutputs = minst.inst.getOutputs().length; //before merge
					minst.inst.mergeMRJobInstruction( instVal );
					minst.addInstructionMetaData(instID, offOutputs, numOutputs);	
				}
				else
				{
					//close current merged instruction
					ret.add(minst); 
					//open new merged instruction
					minst.inst = new MRJobInstruction( instVal );	
					minst.addInstructionMetaData( instID, 0, numOutputs );
				}
			}
		}
		//close last open merged instruction
		ret.add(minst);
		
		//output log info for better understandability for users
		LOG.info("Merged MR-Job instructions: "+workingSet.size()+" --> "+ret.size());
		
		return ret;
	}
}
