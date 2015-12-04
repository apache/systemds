/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import org.apache.sysml.runtime.controlprogram.parfor.opt.Optimizer.PlanInputType;

/**
 * Represents a complete plan of a top-level parfor. This includes the internal
 * representation of the actual current plan as well as additional meta information 
 * that are only kept once per program instead of for each and every plan alternative.
 * 
 */
public class OptTree 
{

	
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
