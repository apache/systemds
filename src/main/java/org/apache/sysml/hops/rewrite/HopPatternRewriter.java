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

import java.util.function.Function;

import org.apache.sysml.hops.Hop;

/**
 * This class is used with HopRewriteRuleWithPatternMatcher to implement the following pattern matching logic:
 * ArrayList<HopPatternRewriter> patternRewriters =  getPatternRewriter();
 * for(HopPatternRewriter patternRewriter : patternRewriters) {
 *   hi = patternRewriter.rewrite(hi);
 * }
 * 
 * Please see org.apache.sysml.hops.rewrite.RewriteGPUSpecificOps class for usage and design documentation.
 */
public class HopPatternRewriter {
	private final HopDagPatternMatcher _matcher;
	private final Function<Hop, Hop> _replacer;
	private final String _name;
	public HopPatternRewriter(String name, HopDagPatternMatcher matcher, Function<Hop, Hop> replacer) {
		_name = name;
		_matcher = matcher;
		_replacer = replacer;
	}
	
	public Hop rewrite(Hop root) {
		boolean printMessage = HopDagPatternMatcher.DEBUG_PATTERNS != null && HopDagPatternMatcher.DEBUG_PATTERNS.contains(_name);
		if(printMessage) {
			HopDagPatternMatcher.DEBUG_REWRITES = true;
			System.out.println("-----------------------------------");
			System.out.println(org.apache.sysml.utils.Explain.explain(root));
		}
		if(_matcher.matches(root)) {
			Hop newHop = _replacer.apply(root);
			if(printMessage) {
				if(newHop == root)
					System.out.println("Initial pattern match for " + _name + " succeeded but replacer returned the same HOP.");
				else
					System.out.println("Pattern match for " + _name + " succeeded.");
				HopDagPatternMatcher.DEBUG_REWRITES = false;
				System.out.println("-----------------------------------");
			}
			return newHop;
		}
		else {
			if(printMessage) {
				System.out.println("Pattern match for " + _name + " failed.");
				HopDagPatternMatcher.DEBUG_REWRITES = false;
				System.out.println("-----------------------------------");
			}
			return root;
		}
	}
}
