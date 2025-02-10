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
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.function.Function;

public class MinimalDifference {
	protected static final Log LOG = LogFactory.getLog(MinimalDifference.class.getName());

	private static RuleContext ctx;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
	}

	@Test
	public void test1() {
		RewriterStatement stmt1 = RewriterUtils.parse("t(t(A))", ctx, "MATRIX:A");
		RewriterStatement stmt2 = RewriterUtils.parse("t(A)", ctx, "MATRIX:A");

		RewriterStatement.MatcherContext mCtx = RewriterStatement.MatcherContext.findMinimalDifference(ctx, stmt2, stmt1);
		stmt1.match(mCtx);
		LOG.info("Minimal Difference: ");
		LOG.info(mCtx.getFirstMismatch()._1.toParsableString(ctx));
		LOG.info(mCtx.getFirstMismatch()._2.toParsableString(ctx));
	}

	@Test
	public void test2() {
		RewriterStatement stmt1 = RewriterUtils.parse("-(A, t(+(A, A)))", ctx, "MATRIX:A");
		RewriterStatement stmt2 = RewriterUtils.parse("-(A, t(*(2, A)))", ctx, "MATRIX:A", "LITERAL_INT:2");

		RewriterStatement.MatcherContext mCtx = RewriterStatement.MatcherContext.findMinimalDifference(ctx, stmt2, stmt1);
		stmt1.match(mCtx);
		LOG.info("Minimal Difference: ");
		LOG.info(mCtx.getFirstMismatch()._1.toParsableString(ctx));
		LOG.info(mCtx.getFirstMismatch()._2.toParsableString(ctx));
	}
}
