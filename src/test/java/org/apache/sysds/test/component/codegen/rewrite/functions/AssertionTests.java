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
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;

public class AssertionTests {
	protected static final Log LOG = LogFactory.getLog(AssertionTests.class.getName());

	private static RuleContext ctx;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
	}

	@Test
	public void test1() {
		RewriterAssertions assertion = new RewriterAssertions(ctx);
		RewriterStatement stmt1 = RewriterUtils.parse("*(*(nrow(A), nrow(B)), *(nrow(C), nrow(A)))", ctx, "MATRIX:A,B,C");
		RewriterStatement nrowA = stmt1.getOperands().get(0).getOperands().get(0);
		RewriterStatement nrowB = stmt1.getOperands().get(0).getOperands().get(1);
		RewriterStatement nrowC = stmt1.getOperands().get(1).getOperands().get(0);
		RewriterStatement nrowA2 = stmt1.getOperands().get(1).getOperands().get(1);

		assert assertion.addEqualityAssertion(nrowA, nrowC, stmt1);
		LOG.info(assertion.getAssertions(nrowA));

		assert !assertion.addEqualityAssertion(nrowA, nrowC, stmt1);
		LOG.info(assertion.getAssertions(nrowC));

		assert assertion.addEqualityAssertion(nrowC, nrowB, stmt1);
		LOG.info(assertion.getAssertions(nrowC));

		LOG.info(assertion.getAssertions(nrowA2));
	}

}
