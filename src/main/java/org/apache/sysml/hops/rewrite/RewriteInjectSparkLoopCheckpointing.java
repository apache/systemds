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

import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.ForStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.parser.WhileStatementBlock;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;

/**
 * Rule: Insert checkpointing operations for caching purposes. Currently, we
 * follow a heuristic of checkpointing (1) all variables used read-only in loops,
 * and (2) intermediates used by multiple consumers.
 * 
 * TODO (2) implement injection for multiple consumers (local and global).
 * 
 */
public class RewriteInjectSparkLoopCheckpointing extends StatementBlockRewriteRule
{
	private boolean _checkCtx = false;
	
	public RewriteInjectSparkLoopCheckpointing(boolean checkParForContext)
	{
		_checkCtx = checkParForContext;
	}
	
	@Override
	public ArrayList<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus status)
		throws HopsException 
	{
		ArrayList<StatementBlock> ret = new ArrayList<StatementBlock>();
		
		if( !OptimizerUtils.isSparkExecutionMode() ) 
		{
			ret.add(sb); // nothing to do here
			return ret; //return original statement block
		}
		
		//1) We currently add checkpoint operations without information about the global program structure,
		//this assumes that redundant checkpointing is prevented at runtime level (instruction-level)
		//2) Also, we do not take size information into account right now. This means that all candidates
		//are checkpointed even if they are only used by CP operations.
		
		int blocksize = status.getBlocksize(); //block size set by reblock rewrite
		
		//apply rewrite for while, for, and parfor (the decision for parfor loop bodies is deferred until parfor
		//optimization because otherwise we would prevent remote parfor)
		if( (sb instanceof WhileStatementBlock || sb instanceof ForStatementBlock)  //incl parfor 
		    && (_checkCtx ? !status.isInParforContext() : true)  )
		{
			//step 1: determine checkpointing candidates
			ArrayList<String> candidates = new ArrayList<String>(); 
			VariableSet read = sb.variablesRead();
			VariableSet updated = sb.variablesUpdated();
			
			for( String rvar : read.getVariableNames() )
				if( !updated.containsVariable(rvar) && read.getVariable(rvar).getDataType()==DataType.MATRIX )
					candidates.add(rvar);
			
			//step 2: insert statement block with checkpointing operations
			if( !candidates.isEmpty() ) //existing candidates
			{
				StatementBlock sb0 = new StatementBlock();
				sb0.setDMLProg(sb.getDMLProg());
				sb0.setAllPositions(sb.getFilename(), sb.getBeginLine(), sb.getBeginColumn(), sb.getEndLine(), sb.getEndColumn());
				ArrayList<Hop> hops = new ArrayList<Hop>();
				VariableSet livein = new VariableSet();
				VariableSet liveout = new VariableSet();
				for( String var : candidates ) 
				{
					DataIdentifier dat = read.getVariable(var);
					DataOp tread = new DataOp(var, DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.TRANSIENTREAD, 
							            dat.getFilename(), dat.getDim1(), dat.getDim2(), dat.getNnz(), blocksize, blocksize);
					tread.setRequiresCheckpoint( true );
					DataOp twrite = new DataOp(var, DataType.MATRIX, ValueType.DOUBLE, tread, DataOpTypes.TRANSIENTWRITE, null);
					HopRewriteUtils.setOutputParameters(twrite, dat.getDim1(), dat.getDim2(), blocksize, blocksize, dat.getNnz());					
					hops.add(twrite);
					livein.addVariable(var, read.getVariable(var));
					liveout.addVariable(var, read.getVariable(var));
				}
				sb0.set_hops(hops);
				sb0.setLiveIn(livein);
				sb0.setLiveOut(liveout);
				ret.add(sb0);
				
				//maintain rewrite status
				status.setInjectedCheckpoints();
			}
		}
			
		//add original statement block to end
		ret.add(sb);
		
		return ret;
	}
}
