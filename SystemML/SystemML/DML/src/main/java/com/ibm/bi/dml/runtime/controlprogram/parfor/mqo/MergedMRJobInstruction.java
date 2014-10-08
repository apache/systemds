/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.mqo;

import java.util.HashMap;
import java.util.LinkedList;

import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.matrix.JobReturn;
import com.ibm.bi.dml.runtime.matrix.MetaData;

/**
 * Merged MR Job instruction to hold the actually merged instruction as well as offsets of
 * result indexes in order to split result meta data after successful execution.
 * 
 */
public class MergedMRJobInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected MRJobInstruction inst;
	protected LinkedList<Long> ids;
	protected HashMap<Long,Integer> outIxOffs;
	protected HashMap<Long,Integer> outIxLens;
	
	public MergedMRJobInstruction()
	{
		ids = new LinkedList<Long>();
		outIxOffs = new HashMap<Long,Integer>();
		outIxLens = new HashMap<Long,Integer>();
	}
	
	public void addInstructionMetaData(long instID, int outIxOffset, int outIxLen)
	{
		ids.add(instID);
		outIxOffs.put(instID, outIxOffset);
		outIxLens.put(instID, outIxLen);
	}
	
	/**
	 * 
	 * @param instID
	 * @param allRet
	 * @return
	 */
	public JobReturn constructJobReturn( long instID, JobReturn retAll )
	{
		//get output offset and len
		int off = outIxOffs.get(instID);
		int len = outIxLens.get(instID);
		
		//create partial output meta data 
		JobReturn ret = new JobReturn();
		ret.successful = retAll.successful;
		ret.metadata = new MetaData[len];
		System.arraycopy(retAll.metadata, off, ret.metadata, 0, len);
		
		return ret;
	}
}
