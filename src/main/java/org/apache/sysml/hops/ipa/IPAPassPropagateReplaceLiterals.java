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

package org.apache.sysml.hops.ipa;

import java.util.ArrayList;

import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.ForStatement;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.FunctionStatement;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.IfStatement;
import org.apache.sysml.parser.IfStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.WhileStatement;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.instructions.cp.ScalarObjectFactory;

/**
 * This rewrite propagates and replaces literals into functions
 * in order to enable subsequent rewrites such as branch removal.
 * 
 */
public class IPAPassPropagateReplaceLiterals extends IPAPass
{
	@Override
	public boolean isApplicable() {
		return InterProceduralAnalysis.PROPAGATE_SCALAR_LITERALS;
	}
	
	@Override
	public void rewriteProgram( DMLProgram prog, FunctionCallGraph fgraph, FunctionCallSizeInfo fcallSizes ) 
		throws HopsException
	{
		for( String fkey : fgraph.getReachableFunctions() ) {
			FunctionOp first = fgraph.getFunctionCalls(fkey).get(0);
			
			//propagate and replace amenable literals into function
			if( fcallSizes.hasSafeLiterals(fkey) ) {
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fkey);
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				ArrayList<DataIdentifier> finputs = fstmt.getInputParams();
				
				//populate call vars with amenable literals
				LocalVariableMap callVars = new LocalVariableMap();
				for( int j=0; j<finputs.size(); j++ )
					if( fcallSizes.isSafeLiteral(fkey, j) ) {
						LiteralOp lit = (LiteralOp) first.getInput().get(j);
						callVars.put(finputs.get(j).getName(), ScalarObjectFactory
								.createScalarObject(lit.getValueType(), lit));
					}
				
				//propagate and replace literals
				for( StatementBlock sb : fstmt.getBody() )
					rReplaceLiterals(sb, callVars);
			}
		}
	}
	
	private void rReplaceLiterals(StatementBlock sb, LocalVariableMap constants) 
		throws HopsException 
	{
		//remove updated literals
		for( String varname : sb.variablesUpdated().getVariableNames() )
			if( constants.keySet().contains(varname) )
				constants.remove(varname);
		
		//propagate and replace literals
		if (sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement ws = (WhileStatement)sb.getStatement(0);
			replaceLiterals(wsb.getPredicateHops(), constants);
			for (StatementBlock current : ws.getBody())
				rReplaceLiterals(current, constants);
		} 
		else if (sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement ifs = (IfStatement) sb.getStatement(0);
			replaceLiterals(isb.getPredicateHops(), constants);
			for (StatementBlock current : ifs.getIfBody())
				rReplaceLiterals(current, constants);
			for (StatementBlock current : ifs.getElseBody())
				rReplaceLiterals(current, constants);
		} 
		else if (sb instanceof ForStatementBlock) {
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fs = (ForStatement)sb.getStatement(0);
			replaceLiterals(fsb.getFromHops(), constants);
			replaceLiterals(fsb.getToHops(), constants);
			replaceLiterals(fsb.getIncrementHops(), constants);
			for (StatementBlock current : fs.getBody())
				rReplaceLiterals(current, constants);
		}
		else {
			replaceLiterals(sb.get_hops(), constants);
		}
	}
	
	private void replaceLiterals(ArrayList<Hop> roots, LocalVariableMap constants) 
		throws HopsException 
	{
		if( roots == null )
			return;
		
		try {
			Hop.resetVisitStatus(roots);
			for( Hop root : roots )
				Recompiler.rReplaceLiterals(root, constants, true);
			Hop.resetVisitStatus(roots);
		}
		catch(Exception ex) {
			throw new HopsException(ex);
		}
	}
	
	private void replaceLiterals(Hop root, LocalVariableMap constants) 
		throws HopsException 
	{
		if( root == null )
			return;
		
		try {
			root.resetVisitStatus();
			Recompiler.rReplaceLiterals(root, constants, true);	
			root.resetVisitStatus();
		}
		catch(Exception ex) {
			throw new HopsException(ex);
		}
	}
}
