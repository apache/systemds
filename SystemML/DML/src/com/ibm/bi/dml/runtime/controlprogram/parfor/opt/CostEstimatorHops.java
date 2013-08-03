package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.utils.DMLRuntimeException;

/**
 * 
 * 
 */
public class CostEstimatorHops extends CostEstimator
{
	public static long DEFAULT_MEM_MR = -1;
	
	private OptTreePlanMappingAbstract _map = null;
	
	static
	{
		DEFAULT_MEM_MR = 20*1024*1024; //20MB
		if( InfrastructureAnalyzer.isLocalMode() )
			DEFAULT_MEM_MR = DEFAULT_MEM_MR + InfrastructureAnalyzer.getRemoteMaxMemorySortBuffer();
	}
	
	
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
		if( value >= DEFAULT_MEM_MR )   	  
		{
			if( h.getExecType()==ExecType.MR ) //CP estimate but MR type
				value = DEFAULT_MEM_MR;
			else if ( h.getExecType()==ExecType.CP && value >= OptimizerUtils.getMemBudget(true) )
			{
				if( DMLScript.rtplatform != DMLScript.RUNTIME_PLATFORM.SINGLE_NODE )
					LOG.warn("Memory estimate larger than budget but CP exec type (op="+h.getOpString()+", name="+h.get_name()+", memest="+h.getMemEstimate()+").");
				value = DEFAULT_MEM_MR;
			}
			//note: if exec type is 'null' lops have never been created (e.g., r(T) for tsmm),
			//in that case, we do not need to raise a warning 
		}
		
		if( h.getForcedExecType()==ExecType.MR ) //forced runtime platform
		{
			value = DEFAULT_MEM_MR;
		}
		
		if( value <= 0 ) //no mem estimate
		{
			LOG.warn("Cannot get memory estimate for hop (op="+h.getOpString()+", name="+h.get_name()+", memest="+h.getMemEstimate()+").");
			value = CostEstimator.DEFAULT_MEM_ESTIMATE_CP;
		}
		
		
		LOG.trace("Memory estimate "+h.get_name()+", "+h.getOpString()+"("+node.getExecType()+")"+"="+OptimizerRuleBased.toMB(value));
		
		return value;
	}

}
