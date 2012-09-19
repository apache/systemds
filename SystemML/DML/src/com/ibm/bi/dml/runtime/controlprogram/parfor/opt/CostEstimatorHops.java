package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * 
 * 
 */
public class CostEstimatorHops extends CostEstimator
{
	public static final double DEFAULT_MEM_MR = 20*1024*1024;
	
	private OptTreePlanMappingAbstract _map = null;
	
	public CostEstimatorHops( OptTreePlanMappingAbstract map )
	{
		_map = map;
	}

	@Override
	public double getLeafNodeEstimate(TestMeasure measure, OptNode node)
		throws DMLRuntimeException 
	{
		if( node.getNodeType() != NodeType.HOP )
			return 0; //generic optnode but no childs (e.g., PB for rmvar inst)
		
		if( measure != TestMeasure.MEMORY_USAGE )
			throw new DMLRuntimeException( "Testmeasure "+measure+" not supported by cost model "+CostModelType.STATIC_MEM_METRIC+"." );
		
		//core mem estimation (use hops estimate)
		Hops h = _map.getMappedHop( node.getID() );
		double value = h.getMemEstimate();
		
		//handle specific cases 
		if(    h.getExecType()==ExecType.MR 
			&& value>DEFAULT_MEM_ESTIMATE_MR ) //CP estimate but MR type
		{
			value = DEFAULT_MEM_MR;
		}
		
		if( value <= 0 ) //no mem estimate
		{
			System.out.println("ParFOR Opt: Warning cannot get memory estimate for hop type "+h.getOpString()+".");
			value = CostEstimator.DEFAULT_MEM_ESTIMATE_CP;
		}
		
		
		//if( OptimizationWrapper.LDEBUG )
		//	System.out.println("ParFOR Opt: Mem estimate "+h.get_name()+", "+h.getOpString()+"="+OptimizerRuleBased.toMB(value));
		
		return value;
	}

}
