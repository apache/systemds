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

import org.apache.sysds.parser.StatementBlock;

import java.util.Arrays;
import java.util.List;

public class RewriteFederatedStatementBlocks extends StatementBlockRewriteRule {

	/**
	 * Indicates if the rewrite potentially splits dags, which is used
	 * for phase ordering of rewrites.
	 *
	 * @return true if dag splits are possible.
	 */
	@Override public boolean createsSplitDag() {
		return false;
	}

	/**
	 * Handle an arbitrary statement block. Specific type constraints have to be ensured
	 * within the individual rewrites. If a rewrite does not apply to individual blocks, it
	 * should simply return the input block.
	 *
	 * @param sb    statement block
	 * @param state program rewrite status
	 * @return list of statement blocks
	 */
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus state) {
		return Arrays.asList(sb);
	}

	/**
	 * Handle a list of statement blocks. Specific type constraints have to be ensured
	 * within the individual rewrites. If a rewrite does not require sequence access, it
	 * should simply return the input list of statement blocks.
	 *
	 * @param sbs   list of statement blocks
	 * @param state program rewrite status
	 * @return list of statement blocks
	 */
	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus state) {
		return sbs;
	}
}
