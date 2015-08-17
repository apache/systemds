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

package com.ibm.bi.dml.hops.rewrite;

import java.util.ArrayList;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.Hop.VisitStatus;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.parser.Expression.ValueType;

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
	
	/**
	 * 
	 * @param hop
	 */
	private void rule_RemoveUnnecessaryCasts( Hop hop )
	{
		//check mark processed
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//recursively process childs
		ArrayList<Hop> inputs = hop.getInput();
		for( int i=0; i<inputs.size(); i++ )
			rule_RemoveUnnecessaryCasts( inputs.get(i) );
		
		//remove cast if unnecessary
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
		
		//mark processed
		hop.setVisited( VisitStatus.DONE );
	}
}
