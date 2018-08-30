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

import org.apache.sysml.hops.Hop;

/**
 * Simple utility class that implements generic structure for HopRewriteRule.
 * Please see org.apache.sysml.hops.rewrite.RewriteGPUSpecificOps class for usage and design documentation.
 */
public abstract class HopRewriteRuleWithPatternMatcher extends HopRewriteRule {
	
	public abstract ArrayList<HopPatternRewriter> getPatternRewriter();
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for( int i = 0; i < roots.size(); i++ )
			applyRules(roots, roots.get(i), false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup) 
		for( int i = 0; i < roots.size(); i++ )
			applyRules(roots, roots.get(i), true );
		Hop.resetVisitStatus(roots, true);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return root;
		
		//one pass rewrite-descend (rewrite created pattern)
		applyRules(null, root, false );
		
		root.resetVisitStatus();
		
		//one pass descend-rewrite (for rollup) 
		applyRules(null, root, true );
		
		return root;
	}
	
	/**
	 * Apply rules
	 * 
	 * @param roots root operators
	 * @param hop high-level operator
	 * @param descendFirst true if recursively process children first
	 */
	private void applyRules(ArrayList<Hop> roots, Hop hop, boolean descendFirst) 
	{
		if(hop.isVisited())
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++) {
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				applyRules(roots, hi, descendFirst); //see below
			
			ArrayList<HopPatternRewriter> patternRewriters =  getPatternRewriter();
			for(HopPatternRewriter patternRewriter : patternRewriters) {
				hi = patternRewriter.rewrite(hi);
			}
			
			if( !descendFirst )
				applyRules(roots, hi, descendFirst);
		}

		hop.setVisited();
	}
}
