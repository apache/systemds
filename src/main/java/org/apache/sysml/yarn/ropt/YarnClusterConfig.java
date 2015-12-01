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

package org.apache.sysml.yarn.ropt;

public class YarnClusterConfig 
{	
		
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
