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

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.Collection;
import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.POptMode;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;


/**
 * Generic optimizer super class that defines the interface of all implemented optimizers.
 * Furthermore it implements basic primitives, used by all optimizers such as the enumeration
 * of plan alternatives and specific rewrites.
 * 
 * Optimization objective: \phi: \min T(prog) | k \leq ck \wedge m(prog) \leq cm 
 *                                      with T(p)=max_(1\leq i\leq k)(T(prog_i). 
 * 
 */
public abstract class Optimizer 
{

	
	protected static final Log LOG = LogFactory.getLog(Optimizer.class.getName());
	
	protected long _numTotalPlans     = -1;
	protected long _numEvaluatedPlans = -1;
	
	public enum PlanInputType {
		ABSTRACT_PLAN,
		RUNTIME_PLAN
	}
	
	public enum CostModelType {
		STATIC_MEM_METRIC,
		RUNTIME_METRICS
	}
	
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
	public abstract boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;	
	
	/**
	 * 
	 * @return
	 */
	public abstract PlanInputType getPlanInputType();
	
	/**
	 * 
	 * @return
	 */
	public abstract CostModelType getCostModelType();
	
	/**
	 * 
	 * @return
	 */
	public abstract POptMode getOptMode();
	
	
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
			Hop hop = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
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
