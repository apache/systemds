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

package org.apache.sysds.lops.rewrite;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.parser.StatementBlock;

import java.util.List;

public class RewriteFixIDs extends LopRewriteRule
{
	@Override
	public List<StatementBlock> rewriteLOPinStatementBlock(StatementBlock sb)
	{
		// Skip if no new Lop nodes are added
		if (!ConfigurationManager.isPrefetchEnabled() && !ConfigurationManager.isBroadcastEnabled()
			&& !ConfigurationManager.isCheckpointEnabled())
			return List.of(sb);

		// Reset the IDs in a depth-first manner
		if (sb.getLops() != null && !sb.getLops().isEmpty()) {
			for (Lop root : sb.getLops())
				assignNewID(root);
			sb.getLops().forEach(Lop::resetVisitStatus);
		}
		return List.of(sb);
	}

	@Override
	public List<StatementBlock> rewriteLOPinStatementBlocks(List<StatementBlock> sbs) {
		return sbs;
	}

	private void assignNewID(Lop lop) {
		if (lop.isVisited())
			return;

		if (lop.getInputs().isEmpty()) {  //leaf node
			lop.setNewID();
			lop.setVisited();
			return;
		}
		for (Lop input : lop.getInputs())
			assignNewID(input);

		lop.setNewID();
		lop.setVisited();
	}
}
