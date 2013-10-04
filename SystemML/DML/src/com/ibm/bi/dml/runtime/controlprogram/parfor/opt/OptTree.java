/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.Optimizer.PlanInputType;

/**
 * Represents a complete plan of a top-level parfor. This includes the internal
 * representation of the actual current plan as well as additional meta information 
 * that are only kept once per program instead of for each and every plan alternative.
 * 
 */
public class OptTree 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	//global contraints 
	private int     _ck;  //max constraint degree of parallelism
	private double  _cm;  //max constraint memory consumption
	
	//actual tree
	private PlanInputType _type = null;
	private OptNode       _root = null;
	
	
	public OptTree( int ck, double cm, OptNode node )
	{
		this( ck, cm, PlanInputType.RUNTIME_PLAN, node );
	}
	
	public OptTree( int ck, double cm, PlanInputType type, OptNode node )
	{
		_ck = ck;
		_cm = cm;
		
		_type = type;
		_root = node;
	}
	
	///////
	// getter and setter
	
	public int getCK()
	{
		return _ck;
	}
	
	public double getCM()
	{
		return _cm;
	}
	
	public PlanInputType getPlanInputType()
	{
		return _type;
	}
	
	public void setPlanInputType( PlanInputType type )
	{
		_type = type;
	}
	
	public OptNode getRoot()
	{
		return _root;
	}
	
	public void setRoot( OptNode n )
	{
		_root = n;
	}
	
	/**
	 * Explain tool: prints the hierarchical plan (including all available 
	 * detail information, if necessary) to <code>stdout</code>.
	 * 
	 * @param withDetails
	 * @return
	 */
	public String explain( boolean withDetails )
	{
		StringBuilder sb = new StringBuilder();
		sb.append("\n");
		sb.append("----------------------------\n");
		sb.append(" EXPLAIN OPT TREE (type=");
		sb.append(_type);
		sb.append(", size=");
		sb.append(_root.size());
		sb.append(")\n");
		sb.append("----------------------------\n");
		sb.append(_root.explain(1, withDetails));
		sb.append("----------------------------\n");
		
		return sb.toString();
	}
}
