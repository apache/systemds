/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;

public class PlanSet 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	 * @return
	 */
	public Plan getPlanWithMinCosts()
	{
		//init global optimal plan and costs
		double optCosts = Double.MAX_VALUE;
		Plan optPlan = null;
		
		//compare costs of all plans
		for( Plan p : _plans )
			if( p.getCosts() < optCosts ) {
				optCosts = p.getCosts();
				optPlan = p;
			}
		
		return optPlan;
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
