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

package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * Rule: RemoveUnnecessaryCasts. For all value type casts check
 * if they are really necessary. If both cast input and output
 * type are the same, the cast itself is redundant. 
 * 
 * There are two use case where this can arise: (1) automatically 
 * inserted casts on function inlining (in case of unknown value 
 * types), and (2) explicit script-level value type casts, that 
 * might be redundant according to the read input data.  
 * 
 * The benefit of this rewrite is negligible for scalars. However,
 * when we support matrices with different value types, those casts
 * might refer to matrices and with that incur large costs.
 * 
 */
public class RewriteRemoveUnnecessaryCasts extends HopRewriteRule
{
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state)
		throws HopsException
	{
		if( roots == null )
			return null;
		
		for( Hop h : roots ) 
			rule_RemoveUnnecessaryCasts( h );
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) 
		throws HopsException
	{
		if( root == null )
			return root;
		
		rule_RemoveUnnecessaryCasts( root );
		
		return root;
	}

	@SuppressWarnings("unchecked")
	private void rule_RemoveUnnecessaryCasts( Hop hop )
	{
		//check mark processed
		if( hop.isVisited() )
			return;
		
		//recursively process childs
		ArrayList<Hop> inputs = hop.getInput();
		for( int i=0; i<inputs.size(); i++ )
			rule_RemoveUnnecessaryCasts( inputs.get(i) );
		
		//remove unnecessary value type cast 
		if( hop instanceof UnaryOp && HopRewriteUtils.isValueTypeCast(((UnaryOp)hop).getOp()) )
		{
			Hop in = hop.getInput().get(0);
			ValueType vtIn = in.getValueType(); //type cast input
			ValueType vtOut = hop.getValueType(); //type cast output
			
			//if input/output types match, no need to cast
			if( vtIn == vtOut && vtIn != ValueType.UNKNOWN ) 
			{
				ArrayList<Hop> parents = hop.getParent();
				for( int i=0; i<parents.size(); i++ ) //for all parents 
				{
					Hop p = parents.get(i);
					ArrayList<Hop> pin = p.getInput();
					for( int j=0; j<pin.size(); j++ ) //for all parent childs
					{
						Hop pinj = pin.get(j);
						if( pinj == hop ) //found parent ref
						{
							//rehang cast input as child of cast consumer
							pin.remove( j ); //remove cast ref
							pin.add(j, in); //add ref to cast input
							in.getParent().remove(hop); //remove cast from cast input parents
							in.getParent().add( p ); //add parent to cast input parents
						}
					}
				}
				parents.clear();	
			}
		}
		
		//remove unnecessary data type casts
		if( hop instanceof UnaryOp && hop.getInput().get(0) instanceof UnaryOp ) {
			UnaryOp uop1 = (UnaryOp) hop;
			UnaryOp uop2 = (UnaryOp) hop.getInput().get(0);
			if( (uop1.getOp()==OpOp1.CAST_AS_MATRIX && uop2.getOp()==OpOp1.CAST_AS_SCALAR) 
				|| (uop1.getOp()==OpOp1.CAST_AS_SCALAR && uop2.getOp()==OpOp1.CAST_AS_MATRIX) ) {
				Hop input = uop2.getInput().get(0);
				//rewire parents
				ArrayList<Hop> parents = (ArrayList<Hop>) hop.getParent().clone();
				for( Hop p : parents )
					HopRewriteUtils.replaceChildReference(p, hop, input);
			}
		}
		
		//mark processed
		hop.setVisited();
	}
}
