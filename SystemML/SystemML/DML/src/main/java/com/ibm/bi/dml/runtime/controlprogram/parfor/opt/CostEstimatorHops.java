/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;

/**
 * 
 * 
 */
public class CostEstimatorHops extends CostEstimator
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
		Hop h = _map.getMappedHop( node.getID() );
		double value = h.getMemEstimate();
		
		//handle specific cases 
		if( value >= DEFAULT_MEM_MR )   	  
		{
			if( h.getExecType()==ExecType.MR ) //CP estimate but MR type
				value = DEFAULT_MEM_MR;
			else if ( h.getExecType()==ExecType.CP && value >= OptimizerUtils.getLocalMemBudget() )
			{
				if( DMLScript.rtplatform != DMLScript.RUNTIME_PLATFORM.SINGLE_NODE && h.getForcedExecType()==null )
					LOG.warn("Memory estimate larger than budget but CP exec type (op="+h.getOpString()+", name="+h.getName()+", memest="+h.getMemEstimate()+").");
				value = DEFAULT_MEM_MR;
			}
			else if ( h.getExecType()==null)
			{
				//note: if exec type is 'null' lops have never been created (e.g., r(T) for tsmm),
				//in that case, we do not need to raise a warning 
				value = DEFAULT_MEM_MR;
			}
		}
		
		if( h.getForcedExecType()==ExecType.MR ) //forced runtime platform
		{
			value = DEFAULT_MEM_MR;
		}
		
		if( value <= 0 ) //no mem estimate
		{
			LOG.warn("Cannot get memory estimate for hop (op="+h.getOpString()+", name="+h.getName()+", memest="+h.getMemEstimate()+").");
			value = CostEstimator.DEFAULT_MEM_ESTIMATE_CP;
		}
		
		
		LOG.trace("Memory estimate "+h.getName()+", "+h.getOpString()+"("+node.getExecType()+")"+"="+OptimizerRuleBased.toMB(value));
		
		return value;
	}

	@Override
	public double getLeafNodeEstimate(TestMeasure measure, OptNode node, ExecType et)
		throws DMLRuntimeException 
	{
		if( node.getNodeType() != NodeType.HOP )
			return 0; //generic optnode but no childs (e.g., PB for rmvar inst)
		
		if( measure != TestMeasure.MEMORY_USAGE )
			throw new DMLRuntimeException( "Testmeasure "+measure+" not supported by cost model "+CostModelType.STATIC_MEM_METRIC+"." );
		
		//core mem estimation (use hops estimate)
		Hop h = _map.getMappedHop( node.getID() );
		double value = h.getMemEstimate();
		if( et != ExecType.CP ) //MR, null
			value = DEFAULT_MEM_MR;
		if( value <= 0 ) //no mem estimate
			value = CostEstimator.DEFAULT_MEM_ESTIMATE_CP;
		
		LOG.trace("Memory estimate (forced exec type) "+h.getName()+", "+h.getOpString()+"("+node.getExecType()+")"+"="+OptimizerRuleBased.toMB(value));
		
		return value;
	}
}
