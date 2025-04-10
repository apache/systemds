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

package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterSearchUtils;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.List;
import java.util.function.Function;

public class SubtreeGeneratorTest {
	protected static final Log LOG = LogFactory.getLog(SubtreeGeneratorTest.class.getName());

	private static RuleContext ctx;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
	}

	@Test
	public void test1() {
		RewriterStatement stmt = RewriterUtils.parse("+(1, a)", ctx, "LITERAL_INT:1", "FLOAT:a");
		List<RewriterStatement> subtrees = RewriterSearchUtils.generateSubtrees(stmt, ctx, 100);

		for (RewriterStatement sub : subtrees) {
			LOG.info("==========");
			LOG.info(sub.toParsableString(ctx, true));
		}

		assert subtrees.size() == 2;
	}

	@Test
	public void test2() {
		RewriterStatement stmt = RewriterUtils.parse("+(+(1, b), a)", ctx, "LITERAL_INT:1", "FLOAT:a,b");
		List<RewriterStatement> subtrees = RewriterSearchUtils.generateSubtrees(stmt, ctx, 100);

		for (RewriterStatement sub : subtrees) {
			LOG.info("==========");
			LOG.info(sub.toParsableString(ctx, true));
		}

		assert subtrees.size() == 3;
	}

	@Test
	public void test3() {
		RewriterStatement stmt = RewriterUtils.parse("-(+(1.0,A),B)", ctx, "LITERAL_FLOAT:1.0", "MATRIX:A,B");
		List<RewriterStatement> subtrees = RewriterSearchUtils.generateSubtrees(stmt, ctx, 100);

		for (RewriterStatement sub : subtrees) {
			LOG.info("==========");
			LOG.info(sub.toParsableString(ctx, true));
		}

		assert subtrees.size() == 3;
	}
}
