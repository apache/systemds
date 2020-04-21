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

import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;

/**
 * Base class for all hop rewrites in order to enable generic
 * application of all rules.
 * 
 */
public abstract class StatementBlockRewriteRule 
{
	protected static final Log LOG = LogFactory.getLog(StatementBlockRewriteRule.class.getName());

	private static final String SB_CUT_PREFIX = "_sbcvar";
	private static final String FUN_CUT_PREFIX = "_funvar";
	private static IDSequence _seq = new IDSequence();
	
	public static String createCutVarName(boolean fun) {
		return fun ?
			FUN_CUT_PREFIX + _seq.getNextID() :
			SB_CUT_PREFIX + _seq.getNextID();
		
	}
	
	/**
	 * Indicates if the rewrite potentially splits dags, which is used
	 * for phase ordering of rewrites.
	 * 
	 * @return true if dag splits are possible.
	 */
	public abstract boolean createsSplitDag();
	
	/**
	 * Handle an arbitrary statement block. Specific type constraints have to be ensured
	 * within the individual rewrites. If a rewrite does not apply to individual blocks, it 
	 * should simply return the input block.
	 * 
	 * @param sb statement block
	 * @param state program rewrite status
	 * @return list of statement blocks
	 */
	public abstract List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state);
	
	/**
	 * Handle a list of statement blocks. Specific type constraints have to be ensured
	 * within the individual rewrites. If a rewrite does not require sequence access, it 
	 * should simply return the input list of statement blocks.
	 * 
	 * @param sbs list of statement blocks
	 * @param state program rewrite status
	 * @return list of statement blocks
	 */
	public abstract List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state);
}
