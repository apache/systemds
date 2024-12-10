package org.apache.sysds.test.component.codegen.rewrite.functions;

import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.hops.rewriter.RuleContext;
import org.junit.BeforeClass;
import org.junit.Test;

public class AssertionTests {

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

		System.out.println(assertion.addEqualityAssertion(nrowA, nrowC, stmt1));
		System.out.println(assertion.getAssertions(nrowA));

		System.out.println(assertion.addEqualityAssertion(nrowA, nrowC, stmt1));
		System.out.println(assertion.getAssertions(nrowC));

		System.out.println(assertion.addEqualityAssertion(nrowC, nrowB, stmt1));
		System.out.println(assertion.getAssertions(nrowC));

		System.out.println(assertion.getAssertions(nrowA2));
	}

}
