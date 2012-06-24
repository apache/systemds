package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.HashMap;
import java.util.LinkedList;

/**
 * TODO: extend this for DP optimizer
 * 
 */
public class MemoTable 
{
	//DP memoization table (nodeID, HashMap<K, V> alternatives)
	private HashMap<Integer,LinkedList<OptNode>> _memo;

	public MemoTable( )
	{
		_memo = new HashMap<Integer, LinkedList<OptNode>>();
	}

	/**
	 * 
	 * @param id
	 * @return
	 */
	public LinkedList<OptNode> getNodes( int id )
	{
		return _memo.get(id);
	}
	
	/**
	 * 
	 * @param id
	 * @param nodelist
	 */
	public void setNodes(int id, LinkedList<OptNode> nodelist)
	{
		_memo.put(id, nodelist);
	}
}
