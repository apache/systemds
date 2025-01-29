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

package org.apache.sysds.hops.rewriter.generated;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.rewrite.HopRewriteRule;
import org.apache.sysds.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysds.hops.rewriter.rule.RewriterRuleSet;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class RewriteAutomaticallyGenerated extends HopRewriteRule {
	public static final String FILE_PATH = null;
	public static RewriteAutomaticallyGenerated existingRewrites;

	private Function<Hop, Hop> rewriteFn;
	public static long totalTimeNanos = 0;
	public static long callCount = 0;
	public static long maxTimeNanos = -1;

	// This constructor could be used to dynamically compile generated rewrite rules from a file
	@Deprecated
	public RewriteAutomaticallyGenerated() {
		// Try to read the file
		try {
			final RuleContext ctx = RewriterUtils.buildDefaultContext();
			List<String> lines = Files.readAllLines(Paths.get(FILE_PATH));
			RewriterRuleSet ruleSet = RewriterRuleSet.deserialize(lines, ctx);

			rewriteFn = ruleSet.compile("AutomaticallyGeneratedRewriteFunction", false);
			existingRewrites = this;
		} catch (IOException e) {
		}
	}

	public RewriteAutomaticallyGenerated(Function<Hop, Hop> rewriteFn) {
		this.rewriteFn = rewriteFn;
	}

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null || rewriteFn == null )
			return roots;

		long startNanos = System.nanoTime();

		//one pass rewrite-descend (rewrite created pattern)
		for( Hop h : roots )
			rule_apply( h, false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup)
		for( Hop h : roots )
			rule_apply( h, true );

		long diff = System.nanoTime() - startNanos;
		totalTimeNanos += diff;
		callCount++;
		if (maxTimeNanos == -1 || maxTimeNanos < diff)
			maxTimeNanos = diff;

		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null || rewriteFn == null )
			return root;

		long startNanos = System.nanoTime();

		//one pass rewrite-descend (rewrite created pattern)
		rule_apply( root, false );

		root.resetVisitStatus();

		//one pass descend-rewrite (for rollup)
		rule_apply( root, true );

		long diff = System.nanoTime() - startNanos;
		totalTimeNanos += diff;
		callCount++;
		if (maxTimeNanos == -1 || maxTimeNanos < diff)
			maxTimeNanos = diff;

		return root;
	}

	private void rule_apply(Hop hop, boolean descendFirst)
	{
		if(hop.isVisited())
			return;

		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);

			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_apply(hi, descendFirst); //see below

			//apply actual simplification rewrites (of childs incl checks)
			hi = rewriteFn.apply(hi);

			//process childs recursively after rewrites (to investigate pattern newly created by rewrites)
			if( !descendFirst )
				rule_apply(hi, descendFirst);
		}

		hop.setVisited();
	}
}
