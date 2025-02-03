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
import org.apache.sysds.hops.rewriter.utils.RewriterSearchUtils;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.apache.sysds.test.component.codegen.rewrite.RewriterTopologySortTests;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

public class RewriterSearchUtilsTest {
	protected static final Log LOG = LogFactory.getLog(RewriterSearchUtilsTest.class.getName());

	private static RuleContext ctx;
	private static Function<RewriterStatement, RewriterStatement> canonicalConverter;

	@BeforeClass
	public static void setup() {
		ctx = RewriterUtils.buildDefaultContext();
		canonicalConverter = RewriterUtils.buildCanonicalFormConverter(ctx, false);
	}

	@Test
	public void testDecode1() {
		int l = 27;
		int n = 5;
		int[] digits = RewriterSearchUtils.fromBaseNNumber(l, n);
		assert digits.length == 3 && digits[0] == 1 && digits[1] == 0 && digits[2] == 2;
	}

	@Test
	public void testDecode2() {
		int l = 5;
		int n = 5;
		int[] digits = RewriterSearchUtils.fromBaseNNumber(l, n);
		LOG.info(Arrays.toString(digits));
		assert digits.length == 2 && digits[0] == 1 && digits[1] == 0;
	}

	@Test
	public void testEncode1() {
		int[] digits = new int[] { 1, 0, 2 };
		int[] digits2 = new int[] {4, 4, 4};
		int n = 5;
		int l = RewriterSearchUtils.toBaseNNumber(digits, n);
		int l2 = RewriterSearchUtils.toBaseNNumber(digits2, n);
		LOG.info(l);
		LOG.info(Integer.toBinaryString(l));
		LOG.info(l2);
		LOG.info(Integer.toBinaryString(l2));
		assert l == 27;
	}

	@Test
	public void testRandomStatementGeneration() {
		LOG.info(RewriterSearchUtils.getMaxSearchNumberForNumOps(3));
		int ctr = 0;
		for (int i = 0; i < 20; i++) {
			List<RewriterSearchUtils.Operand> ops = RewriterSearchUtils.decodeOrderedStatements(i);
			//LOG.info("Idx: " + i);
			//LOG.info(ops);
			//LOG.info(RewriterAlphabetEncoder.buildAllPossibleDAGs(ops, ctx, false).size());
			for (RewriterStatement stmt : RewriterSearchUtils.buildAllPossibleDAGs(ops, ctx, true)) {
				LOG.info("Base: " + stmt.toParsableString(ctx));
				for (RewriterStatement sstmt : RewriterSearchUtils.buildAssertionVariations(stmt, ctx)) {
					canonicalConverter.apply(sstmt);
					LOG.info(sstmt.toParsableString(ctx));
					//LOG.info("Raw: " + sstmt);
					ctr++;
				}
			}
		}

		LOG.info("Total DAGs: " + ctr);
	}

	@Test
	public void testRandomStatementGeneration2() {
		int ctr = 0;
		//for (int i = 0; i < 20; i++) {
			List<RewriterSearchUtils.Operand> ops = List.of(RewriterSearchUtils.instructionAlphabet[3], RewriterSearchUtils.instructionAlphabet[16], RewriterSearchUtils.instructionAlphabet[6]);
			//LOG.info("Idx: " + i);
			//LOG.info(ops);
			//LOG.info(RewriterAlphabetEncoder.buildAllPossibleDAGs(ops, ctx, false).size());
			for (RewriterStatement stmt : RewriterSearchUtils.buildAllPossibleDAGs(ops, ctx, true)) {
				LOG.info("Base: " + stmt.toParsableString(ctx));
				for (RewriterStatement sstmt : RewriterSearchUtils.buildVariations(stmt, ctx)) {
					canonicalConverter.apply(sstmt);
					LOG.info(sstmt.toParsableString(ctx));
					//LOG.info("Raw: " + sstmt);
					ctr++;
				}
			}
		//}

		LOG.info("Total DAGs: " + ctr);
	}

	@Test
	public void test() {
		RewriterStatement stmt = RewriterUtils.parse("+([](A, 1, 1, 1, 1), B)", ctx, "MATRIX:A,B", "LITERAL_INT:1");
		stmt = canonicalConverter.apply(stmt);
		LOG.info(stmt.toParsableString(ctx));
	}

}
