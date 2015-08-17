/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;

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
	
	/**
	 * 
	 * @param steps
	 */
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
	
	/**
	 * 
	 * @param pbs
	 * @param mem
	 * @throws HopsException
	 */
	private void getMemoryEstimates( ArrayList<ProgramBlock> pbs, ArrayList<Long> mem ) 
		throws HopsException
	{
		for( ProgramBlock pb : pbs )
			getMemoryEstimates(pb, mem);
	}
	
	/**
	 * 
	 * @param pb
	 * @param mem
	 * @throws HopsException
	 */
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
	
	/**
	 * 
	 * @param hop
	 * @param mem
	 */
	private void getMemoryEstimates( Hop hop, ArrayList<Long> mem )
	{
		if( hop.getVisited() == Hop.VisitStatus.DONE )
			return;

		//process childs
		for(Hop hi : hop.getInput())
			getMemoryEstimates(hi, mem);
		
		//add memory estimates (scaled by CP memory ratio)
		mem.add( (long)( (hop.getMemEstimate()+DEFAULT_MEM_ADD)
				          /OptimizerUtils.MEM_UTIL_FACTOR) );
		
		hop.setVisited(Hop.VisitStatus.DONE);
	}
	
}
