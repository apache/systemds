/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

public class YarnClusterConfig 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private long _minAllocMB = -1;
	private long _maxAllocMB = -1;
	private long _numNodes = -1;
	private long _numCores = -1;
	
	public YarnClusterConfig()
	{
		_minAllocMB = -1;
		_maxAllocMB = -1;
		_numNodes = -1;
		_numCores = -1;
	}
	
	public long getMinAllocationMB()
	{
		return _minAllocMB;
	}
	
	public void setMinAllocationMB(long min)
	{
		_minAllocMB = min;
	}
	
	public long getMaxAllocationMB()
	{
		return _maxAllocMB;
	}
	
	public void setMaxAllocationMB(long max)
	{
		_maxAllocMB = max;
	}
	
	public long getNumCores()
	{
		return _numCores;
	}
	
	public void setNumCores(long ncores)
	{
		_numCores = ncores;
	}
	
	public long getNumNodes()
	{
		return _numNodes;
	}
	
	public void setNumNodes(long nnodes)
	{
		_numNodes = nnodes;
	}
	
	public long getAvgNumCores()
	{
		return (long)Math.round((double)_numCores / _numNodes);
	}
}
