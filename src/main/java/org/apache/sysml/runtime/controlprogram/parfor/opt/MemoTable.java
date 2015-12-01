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
			if( !entries.isEmpty() )
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
			if( entries != null && !entries.isEmpty() )
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
			if( entries != null && !entries.isEmpty() )
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
