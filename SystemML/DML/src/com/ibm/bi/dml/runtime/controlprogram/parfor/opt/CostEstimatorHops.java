package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * 
 * 
 */
public class CostEstimatorHops extends CostEstimator
{
	private OptTreePlanMappingAbstract _map = null;
	
	public CostEstimatorHops( OptTreePlanMappingAbstract map )
	{
		_map = map;
	}

	@Override
	public double getLeafNodeEstimate(TestMeasure measure, OptNode node)
		throws DMLRuntimeException 
	{
		if( measure != TestMeasure.MEMORY_USAGE )
			throw new DMLRuntimeException( "Testmeasure "+measure+" not supported by cost model "+CostModelType.STATIC_MEM_METRIC+"." );
		
		Hops h = _map.getMappedHop( node.getID() );
		double value = h.getMemEstimate();
		if( value <= 0 )
		{
			System.out.println("ParFOR Opt: Warning cannot get memory estimate for hop type "+h.getOpString()+".");
			value = CostEstimator.DEFAULT_MEM_ESTIMATE;
		}
		
		return value;
	}

}
