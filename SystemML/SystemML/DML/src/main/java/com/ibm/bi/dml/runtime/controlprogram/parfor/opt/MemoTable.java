/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;

/**
 * 
 * 
 */
public class MemoTable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	//logical plan node, list of physical plan nodes
	private Map<Long,Collection<MemoTableEntry>> _memo; 

	public MemoTable( )
	{
		_memo = new HashMap<Long, Collection<MemoTableEntry>>();
	}

	/**
	 * 
	 * @param ID
	 * @param e
	 * @param keepOnlyMin
	 */
	public void putMemoTableEntry( long ID, MemoTableEntry e, boolean keepOnlyMin )
	{
		//create memo structure on demand
		Collection<MemoTableEntry> entries = _memo.get(ID);
		if( entries == null )
		{
			entries = new LinkedList<MemoTableEntry>();
			_memo.put(ID, entries);
		}
		
		//add the respective entry
		if( keepOnlyMin )
		{
			if( entries.size()>0 )
			{
				MemoTableEntry old = entries.iterator().next(); 
				if( e.getEstLTime()<old.getEstLTime() )
				{
					entries.remove(old);
					entries.add(e);
				}
			}
			else
				entries.add(e);
		}
		else
			entries.add(e);
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean hasCandidates()
	{
		for( Collection<MemoTableEntry> entries : _memo.values() )
			if( entries != null && entries.size()>0 )
				return true;
		return false;	
	}
	
	/**
	 * 
	 * @return
	 */
	public Collection<MemoTableEntry> getCandidates()
	{
		Collection<MemoTableEntry> C = new LinkedList<MemoTableEntry>();
		
		for( Collection<MemoTableEntry> entries : _memo.values() )
			if( entries != null && entries.size()>0 )
				C.addAll(entries);
		
		return C;	
	}
	
	/**
	 * 
	 * @param ID
	 * @return
	 */
	public MemoTableEntry getMinTimeEntry( long ID )
	{
		return getMin( ID, TestMeasure.EXEC_TIME );
	}
	
	/**
	 * 
	 * @param ID
	 * @return
	 */
	public MemoTableEntry getMinMemEntry( long ID )
	{
		return getMin( ID, TestMeasure.MEMORY_USAGE );
	}
	
	/**
	 * 
	 * @param ID
	 * @param measure
	 * @return
	 */
	private MemoTableEntry getMin( long ID, TestMeasure measure )
	{
		MemoTableEntry minObj = null;
		double minVal = Double.MAX_VALUE;
		
		Collection<MemoTableEntry> entries = _memo.get( ID );
		if( entries != null )
			for( MemoTableEntry e : entries )
				switch( measure )
				{
					case EXEC_TIME:
						if( e.getEstLTime() < minVal )
						{
							minObj = e;
							minVal = e.getEstLTime();
						}
						break;
					case MEMORY_USAGE:
						if( e.getEstLMemory() < minVal )
						{
							minObj = e;
							minVal = e.getEstLMemory();
						}
						break;
				}
			
		return minObj;
	}

}
