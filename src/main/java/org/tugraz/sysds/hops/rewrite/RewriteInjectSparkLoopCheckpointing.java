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

package org.tugraz.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.parser.DataIdentifier;
import org.tugraz.sysds.parser.ForStatementBlock;
import org.tugraz.sysds.parser.IndexedIdentifier;
import org.tugraz.sysds.parser.StatementBlock;
import org.tugraz.sysds.parser.VariableSet;
import org.tugraz.sysds.parser.WhileStatementBlock;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.common.Types.ValueType;

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
	
	public RewriteInjectSparkLoopCheckpointing(boolean checkParForContext) {
		_checkCtx = checkParForContext;
	}
	
	@Override
	public boolean createsSplitDag() {
		return true;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus status)
	{
		if( !OptimizerUtils.isSparkExecutionMode() ) {
			// nothing to do here, return original statement block
			return Arrays.asList(sb);
		}
		
		//1) We currently add checkpoint operations without information about the global program structure,
		//this assumes that redundant checkpointing is prevented at runtime level (instruction-level)
		//2) Also, we do not take size information into account right now. This means that all candidates
		//are checkpointed even if they are only used by CP operations.
		
		ArrayList<StatementBlock> ret = new ArrayList<>();
		int blocksize = status.getBlocksize(); //block size set by reblock rewrite
		
		//apply rewrite for while, for, and parfor (the decision for parfor loop bodies is deferred until parfor
		//optimization because otherwise we would prevent remote parfor)
		if( (sb instanceof WhileStatementBlock || sb instanceof ForStatementBlock)  //incl parfor 
		    && (_checkCtx ? !status.isInParforContext() : true)  )
		{
			//step 1: determine checkpointing candidates
			ArrayList<String> candidates = new ArrayList<>();
			VariableSet read = sb.variablesRead();
			VariableSet updated = sb.variablesUpdated();
			
			for( String rvar : read.getVariableNames() )
				if( !updated.containsVariable(rvar) && (read.getVariable(rvar).getDataType()==DataType.MATRIX ||
						read.getVariable(rvar).getDataType()==DataType.TENSOR))
					candidates.add(rvar);
			
			//step 2: insert statement block with checkpointing operations
			if( !candidates.isEmpty() ) //existing candidates
			{
				StatementBlock sb0 = new StatementBlock();
				sb0.setDMLProg(sb.getDMLProg());
				sb0.setParseInfo(sb);
				ArrayList<Hop> hops = new ArrayList<>();
				VariableSet livein = new VariableSet();
				VariableSet liveout = new VariableSet();
				for( String var : candidates ) 
				{
					DataIdentifier dat = read.getVariable(var);
					long dim1 = (dat instanceof IndexedIdentifier) ? ((IndexedIdentifier)dat).getOrigDim1() : dat.getDim1();
					long dim2 = (dat instanceof IndexedIdentifier) ? ((IndexedIdentifier)dat).getOrigDim2() : dat.getDim2();
					DataOp tread = new DataOp(var, DataType.MATRIX, ValueType.FP64, DataOpTypes.TRANSIENTREAD, 
						dat.getFilename(), dim1, dim2, dat.getNnz(), blocksize);
					tread.setRequiresCheckpoint(true);
					DataOp twrite = HopRewriteUtils.createTransientWrite(var, tread);
					hops.add(twrite);
					livein.addVariable(var, read.getVariable(var));
					liveout.addVariable(var, read.getVariable(var));
				}
				sb0.setHops(hops);
				sb0.setLiveIn(livein);
				sb0.setLiveOut(liveout);
				sb0.setSplitDag(true);
				ret.add(sb0);
				
				//maintain rewrite status
				status.setInjectedCheckpoints();
			}
		}
			
		//add original statement block to end
		ret.add(sb);
		
		return ret;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}
}
