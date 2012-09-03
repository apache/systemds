package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.Collection;
import java.util.LinkedList;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


/**
 * Generic optimizer super class that defines the interface of all implemented optimizers.
 * Furthermore it implements basic primitives, used by all optimizers such as the enumeration
 * of plan alterantives and specific rewrites.
 * 
 * Optimization objective: \phi: \min T(prog) | k \leq ck \wedge m(prog) \leq cm 
 *                                      with T(p)=max_(1\leq i\leq k)(T(prog_i). 
 * 
 * Known implementation classes: OptimizerHeuristic (time: O(m)), OptimizerGreedyEnum 
 * (time: O(m^2)), and OptimizerDPEnum (time: O(2^m)) 
 * 
 * 
 * CURRENT LIST of REWRITES
 * - Select execution strategy (parfor, hops that allow CP/MR)				done
 * - Nested parallelism (ParFOR[Remote] -> ParFOR[Remote],ParFOR[Local])	done
 * - Select task partitioner												done
 * - Select degree of parallelism for hierarchy of PB/inst					done
 * - Remove unnecessary nested parallelism (ParFOR[k=1]->FOR)				done
 * 
 * Experimental
 * - Select data partitioning, recompile, and indexed read                  ? (runtime support)
 * - Choose data partitioning granularity
 * 
 * 
 * TODO to be implemented
 * - result merge aware parallelization
 * - Select degree/distribution of parallelism for MR jobs (#mappers,#reducers)
 * - Choose configuration properties (sort.io, etc)
 * - Rewrite blocked matrix/file access (sharing-aware)
 * - Redundant execution (whenever compute is cheaper than transfer)
 * - Split single ParFOR into ParFOR of different types wrt data transfer and inst
 * - Rewrite specific CP inst (e.g., matmult) to ParFOR (inst on independent blocks)
 * - ...  
 * 
 * 
 * NOTE: better statistics, holistic view on all included instructions, awareness of par and mem, 
 */
public abstract class Optimizer 
{
	protected long _numTotalPlans     = -1;
	protected long _numEvaluatedPlans = -1;
	
	protected Optimizer()
	{
		_numTotalPlans     = 0;
		_numEvaluatedPlans = 0;
	}
	
	/**
	 * 
	 * @param plan
	 * @return true if plan changed, false otherwise
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public abstract boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;	
	
	///////
	//methods for evaluating the overall properties and costing  

	/**
	 *
	 * @return
	 */
	public long getNumTotalPlans()
	{
		return _numTotalPlans;
	}
	
	/**
	 * 
	 * @return
	 */
	public long getNumEvaluatedPlans()
	{
		return _numEvaluatedPlans;
	}
	
	
	///////
	//methods for common basic primitives
	
	/**
	 * Enum node plans (only for current opt node)
	 */
	protected Collection<OptNode> enumPlans( OptNode n, double lck )
	{
		Collection<OptNode> plans = enumerateExecTypes( n );
		
		//TODO additional enumerations / potential rewrites go here
			
		return plans;
	}

	/**
	 * 
	 * @param n
	 * @return
	 */
	private Collection<OptNode> enumerateExecTypes( OptNode n )
	{
		Collection<OptNode> dTypes = new LinkedList<OptNode>();
		boolean genAlternatives = false;
		
		//determine if alternatives should be generated
		if( n.isLeaf() ) //hop
		{
			Hops hop = OptTreeConverter.getHLObjectMapping().getMappedHop(n.getID());
			if( hop.allowsAllExecTypes() )
				genAlternatives = true;
		}
		else if( n.getNodeType()==NodeType.PARFOR ) //parfor pb
		{
			genAlternatives = true;
		}

		//generate alternatives
		if( genAlternatives )
		{
			OptNode c1 = n.createShallowClone();
			OptNode c2 = n.createShallowClone();
			c1.setExecType(ExecType.CP);
			c2.setExecType(ExecType.MR);
			dTypes.add( c1 );
			dTypes.add( c2 );
		}
		
		return dTypes;	
	}
}
