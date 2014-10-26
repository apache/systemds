/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final int DEFAULT_NSTEPS = 20;

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
		long gap = (long)(_max - _min) / (_nsteps-1); //MB granularity
		
		//enumerate bins between equi grid points
		HashMap<Long, Integer> map = new HashMap<Long,Integer>();
		long v = _min;
		for (int i = 0; i < _nsteps; i++) {
			map.put( getKey(_min, gap, v), 0 );
			v += gap;
		}
		
		//get memory estimates
		ArrayList<Long> mem = new ArrayList<Long>();
		getMemoryEstimates( _prog, mem );
		for( Long est : mem ){
			Integer cnt = map.get( getKey(_min, gap, est));
			cnt = (cnt==null) ? 0 : cnt;
			map.put(getKey(_min, gap, est), cnt+1);
		}
		
		//prepare output (for disjointness)
		HashSet<Long> preRet = new HashSet<Long>();
		for( Entry<Long, Integer> e : map.entrySet() ){
			if( e.getValue() > 0 ){
				preRet.add(_min+e.getKey()*gap);
				if( e.getKey()>0 )
					preRet.add(_min+(e.getKey()+1)*gap);
			}
		}
		
		//create sorted output (to prevent over-provisioning)
		for( Long val : preRet )
			if( val <= _max )//filter max
				ret.add(val);
		Collections.sort(ret); //asc
		
		return ret;
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
		if( hop.get_visited() == Hop.VISIT_STATUS.DONE )
			return;

		//process childs
		for(Hop hi : hop.getInput())
			getMemoryEstimates(hi, mem);
		
		//add memory estimates (scaled by CP memory ratio)
		mem.add( (long)(hop.getMemEstimate()/OptimizerUtils.MEM_UTIL_FACTOR) );
		
		hop.set_visited(Hop.VISIT_STATUS.DONE);
	}
	
	/**
	 * 
	 * @param min
	 * @param gap
	 * @param val
	 * @return
	 */
	private static long getKey( long min, long gap, long val )
	{
		return Math.max( (val-min)/gap, 0 );
	}
	
	
	/*
	 	public int countInterestPoints(double min, double max, ArrayList<Double> interested) {
		int count = 0;
		for (Double v : interested) {
			if (v < min)
				break;
			if (v <= max)
				count++;
		}
		return count;
	}

	public ArrayList<Double> genHybridGrid(double min, double max, int mainStep, int subStep, ArrayList<Double> interested) {
		int i, j;
		ArrayList<Double> ret = new ArrayList<Double> ();
		
		double mainGap = (max - min) / mainStep;
		double subGap = mainGap / subStep;
		
		double current = min;
		ret.add(current);
		for (i = 0; i < mainStep; i++) {
			if (countInterestPoints(current, current + mainGap, interested) > 0) {
				double tmp = 0;
				for (j = 0; j < subStep - 1; j++) {
					tmp += subGap;
					ret.add(current + tmp);
				}
			}
			current += mainGap;
			if (i + 1 == mainStep)
				ret.add(max);	// just to make sure the last sample is exactly "max"
			else
				ret.add(current);
		}
		return ret;
	}
	 */
}
