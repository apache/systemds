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

package org.apache.sysds.hops.rewriter;

import java.util.HashMap;

public class RewriterStatementEntry {
	private final RuleContext ctx;
	final RewriterStatement instr;

	public RewriterStatementEntry(final RuleContext ctx, RewriterStatement instr) {
		this.ctx = ctx;
		this.instr = instr;
	}

	@Override
	public int hashCode() {
		return instr.structuralHashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof RewriterStatement) {
			if (instr == o)
				return true;
			if (instr.structuralHashCode() != ((RewriterStatement)o).structuralHashCode())
				return false;
			return instr.match(new RewriterStatement.MatcherContext(ctx, (RewriterStatement) o, new RewriterStatement.RewriterPredecessor(), (RewriterStatement) o, instr, false, false, false, false, false, false, true, false, false, false, new HashMap<>()));
		}

		if (o.hashCode() != hashCode())
			return false;

		if (o instanceof RewriterStatementEntry) {
			if (instr == ((RewriterStatementEntry) o).instr)
				return true;
			return instr.match(new RewriterStatement.MatcherContext(ctx, ((RewriterStatementEntry) o).instr, new RewriterStatement.RewriterPredecessor(), ((RewriterStatementEntry) o).instr, instr, false, false, false, false, false, false, true, false, false, false, new HashMap<>()));
		}
		return false;
	}
}
