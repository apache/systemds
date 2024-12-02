package org.apache.sysds.hops.rewriter.assertions;

import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

public class RewriterAssertionUtils {
	public static RewriterAssertions buildImplicitAssertions(RewriterStatement root, final RuleContext ctx) {
		RewriterAssertions assertions = new RewriterAssertions(ctx);
		buildImplicitAssertions(root, assertions, ctx);
		return assertions;
	}

	public static void buildImplicitAssertions(RewriterStatement root, RewriterAssertions assertions, final RuleContext ctx) {
		root.forEachPreOrder(cur -> {
			buildImplicitAssertion(cur, assertions, ctx);
			return true;
		}, false);
	}

	public static boolean buildImplicitAssertion(RewriterStatement stmt, RewriterAssertions assertions, final RuleContext ctx) {
		if (!stmt.isInstruction())
			return false;

		switch (stmt.trueInstruction()) {
			case "%*%":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNRow());
				return true;
			case "diag":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(0).getNRow());
				return true;
			case "RBind":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNCol());
				return true;
			case "CBind":
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(1).getNRow());
				return true;
			case "1-*":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNCol());
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(1).getNRow());
				return true;
			case "+*":
			case "-*":
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(2).getNCol());
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(2).getNRow());
				return true;
		}

		switch (stmt.trueTypedInstruction(ctx)) {
			case "trace(MATRIX)":
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(0).getNCol());
				return true;
			case "cast.FLOAT(MATRIX)":
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(0).getNCol());
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), RewriterStatement.literal(ctx, 1L));
				return true;
		}

		if (((RewriterInstruction) stmt).hasProperty("ElementWiseInstruction", ctx)) {
			if (stmt.getChild(0).getResultingDataType(ctx).equals("MATRIX")
				&& stmt.getChild(1).getResultingDataType(ctx).equals("MATRIX")) {
				assertions.addEqualityAssertion(stmt.getChild(0).getNCol(), stmt.getChild(1).getNCol());
				assertions.addEqualityAssertion(stmt.getChild(0).getNRow(), stmt.getChild(1).getNRow());
				return true;
			}
		}

		return false;
	}
}
