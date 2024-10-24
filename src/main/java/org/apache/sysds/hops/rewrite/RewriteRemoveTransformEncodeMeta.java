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

package org.apache.sysds.hops.rewrite;

import java.util.Arrays;
import java.util.List;

import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.parser.StatementBlock;

/**
 * Rule: If transformencode procudes a meta data frame which is never
 * used, flag transformencode to never allocate an serialize this frame.
 */
public class RewriteRemoveTransformEncodeMeta extends StatementBlockRewriteRule
{
	private final static String TF_OPCODE = "TRANSFORMENCODE";
		
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state)
	{
		if( sb.getHops() == null || sb.getHops().isEmpty() )
			return Arrays.asList(sb);
		
		//Transformencode is a multi-return FunctionOp and always appears as root
		//of the DAG. We then check that the meta data object is never used,
		//that is, the meta data is not in the live-out variables of the statementblock
		Hop root = sb.getHops().get(0);
		if( root instanceof FunctionOp
			&& TF_OPCODE.equals(((FunctionOp)root).getFunctionName()) )
		{
			FunctionOp func = (FunctionOp)root;
			if( !sb.liveOut().containsVariable(func.getOutputVariableNames()[1])
				&& func.getInput().size() == 2) { //not added yet
				func.getInput().add(new LiteralOp(false));
				LOG.debug("Applied removeTransformEncodeMeta (line "+ func.getBeginLine() +").");
			}
		}
		
		return Arrays.asList(sb);
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}

	@Override
	public boolean createsSplitDag() {
		return false;
	}
}
