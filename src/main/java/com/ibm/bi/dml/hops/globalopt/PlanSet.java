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

package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFCrossBlockNode;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode;
import com.ibm.bi.dml.hops.globalopt.gdfgraph.GDFNode.NodeType;

public class PlanSet 
{
	
	private ArrayList<Plan> _plans = null;
	
	public PlanSet()
	{
		_plans = new ArrayList<Plan>();
	}

	public PlanSet(ArrayList<Plan> plans)
	{
		_plans = plans;
	}
	
	public ArrayList<Plan> getPlans()
	{
		return _plans;
	}
	
	public void setPlans(ArrayList<Plan> plans)
	{
		_plans = plans;
	}
	
	public int size()
	{
		if( _plans == null )
			return 0;
		
		return _plans.size();
	}
	
	public boolean isEmpty()
	{
		if( _plans == null )
			return true;
		
		return _plans.isEmpty();	
	}

	/**
	 * 
	 * @param pc
	 * @return
	 */
	public PlanSet crossProductChild(PlanSet pc) 
	{
		//check for empty child plan set (return current plan set)
		if( pc==null || pc.isEmpty() ) {
			return this;
		}
		//check for empty parent plan set (pass-through child)
		//(e.g., for crossblockop; this also implicitly reused costed runtime plans)
		if( _plans == null || _plans.isEmpty() ) {
			return pc;
		}
		
		ArrayList<Plan> Pnew = new ArrayList<Plan>();  
		
		// create cross product of plansets between partial and child plans
		for( Plan p : _plans )
			for( Plan c : pc.getPlans() )
			{
				Plan pnew = new Plan(p);
				pnew.addChild( c );
				Pnew.add( pnew );
			}
		
		return new PlanSet(Pnew);
	}

	/**
	 * 
	 * @param node
	 * @return
	 */
	public PlanSet selectChild( GDFNode node )
	{
		String varname = (node.getNodeType()==NodeType.HOP_NODE) ? node.getHop().getName() :
			            ((GDFCrossBlockNode)node).getName();
		
		ArrayList<Plan> Pnew = new ArrayList<Plan>();  
		for( Plan p : _plans )
			if( p.getNode().getHop()!=null 
			   &&p.getNode().getHop().getName().equals(varname) )
			{
				Pnew.add( p );
			}
		
		return new PlanSet(Pnew);
	}
	
	/**
	 * 
	 * @param ps
	 * @return
	 */
	public PlanSet union( PlanSet ps )
	{
		ArrayList<Plan> Pnew = new ArrayList<Plan>(_plans);  
		for( Plan p : ps._plans )
			Pnew.add( p );
		
		return new PlanSet(Pnew);
	}
	
	/**
	 * 
	 * @return
	 */
	public Plan getPlanWithMinCosts()
	{
		//init global optimal plan and costs
		double optCosts = Double.MAX_VALUE;
		Plan optPlan = null;
		
		//compare costs of all plans
		for( Plan p : _plans ) {
			if( p.getCosts() < optCosts ) {
				optCosts = p.getCosts();
				optPlan = p;
			}
		}	
		
		return optPlan;
	}
	
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("PLAN_SET@"+super.hashCode()+":\n");
		for( Plan p : _plans ) {
			sb.append("--");
			sb.append( p.toString() );
			sb.append("\n");
		}
		
		return sb.toString();
	}
}
