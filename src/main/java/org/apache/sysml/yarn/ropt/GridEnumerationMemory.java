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

package org.apache.sysml.yarn.ropt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;

public class GridEnumerationMemory extends GridEnumeration
{
	
	public static final int DEFAULT_NSTEPS = 20; //old config: 15
	public static final int DEFAULT_MEM_ADD = 1*1024*1024; //1MB
	
	private int _nsteps = -1;
	
	public GridEnumerationMemory( ArrayList<ProgramBlock> prog, long min, long max ) 
		throws DMLRuntimeException
	{
		super(prog, min, max);
		
		_nsteps = DEFAULT_NSTEPS;
	}
	
	public void setNumSteps( int steps )
	{
		_nsteps = steps;
	}
	
	@Override
	public ArrayList<Long> enumerateGridPoints() 
		throws DMLRuntimeException, HopsException
	{
		ArrayList<Long> ret = new ArrayList<Long>();
		long gap = (long)(_max - _min) / (_nsteps-1);
		
		//get memory estimates
		ArrayList<Long> mem = new ArrayList<Long>();
		getMemoryEstimates( _prog, mem );
		
		//binning memory estimates to equi grid 
		HashSet<Long> preRet = new HashSet<Long>();
		for( Long val : mem )
		{
			if( val < _min ) 
				preRet.add( _min ); //only right side
			else if( val > _max )
				preRet.add( _max ); //only left side
			else
			{
				long bin = Math.max((val-_min)/gap,0);				
				preRet.add( filterMax(_min + bin*gap) );
				preRet.add( filterMax(_min + (bin+1)*gap) );
			}
		}
		
		//create sorted output (to prevent over-provisioning)
		for( Long val : preRet )
			ret.add(val);
		Collections.sort(ret); //asc
		
		return ret;
	}
	
	private long filterMax( long val )
	{
		if( val > _max ) //truncate max
			return _max;
		
		return val;
	}
	
	private void getMemoryEstimates( ArrayList<ProgramBlock> pbs, ArrayList<Long> mem ) 
		throws HopsException
	{
		for( ProgramBlock pb : pbs )
			getMemoryEstimates(pb, mem);
	}
	
	private void getMemoryEstimates( ProgramBlock pb, ArrayList<Long> mem ) 
		throws HopsException
	{
		if (pb instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocks(), mem);
		}
		else if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock fpb = (WhileProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocks(), mem);
		}	
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock fpb = (IfProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocksIfBody(), mem);
			getMemoryEstimates(fpb.getChildBlocksElseBody(), mem);
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			ForProgramBlock fpb = (ForProgramBlock)pb;
			getMemoryEstimates(fpb.getChildBlocks(), mem);
		}
		else
		{
			StatementBlock sb = pb.getStatementBlock();
			if( sb != null && sb.get_hops() != null ){
				Hop.resetVisitStatus(sb.get_hops());
				for( Hop hop : sb.get_hops() )
					getMemoryEstimates(hop, mem);
			}
		}
	}
	
	private void getMemoryEstimates( Hop hop, ArrayList<Long> mem )
	{
		if( hop.isVisited() )
			return;

		//process childs
		for(Hop hi : hop.getInput())
			getMemoryEstimates(hi, mem);
		
		//add memory estimates (scaled by CP memory ratio)
		mem.add( (long)( (hop.getMemEstimate()+DEFAULT_MEM_ADD)
				          /OptimizerUtils.MEM_UTIL_FACTOR) );
		
		hop.setVisited();
	}
	
}
