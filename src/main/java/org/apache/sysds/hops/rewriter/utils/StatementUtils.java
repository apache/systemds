package org.apache.sysds.hops.rewriter.utils;

import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

public class StatementUtils {
	public static RewriterStatement max(final RuleContext ctx, RewriterStatement... of) {
		if (of.length == 1)
			return of[0];

		if (of.length == 2)
			return new RewriterInstruction("max", ctx, of);

		throw new UnsupportedOperationException();
	}

	public static RewriterStatement min(final RuleContext ctx, RewriterStatement... of) {
		if (of.length == 1)
			return of[0];

		if (of.length == 2)
			return new RewriterInstruction("min", ctx, of);

		throw new UnsupportedOperationException();
	}

	public static RewriterStatement length(final RuleContext ctx, RewriterStatement matrix) {
		if (!matrix.getResultingDataType(ctx).equals("MATRIX"))
			throw new IllegalArgumentException();

		return new RewriterInstruction("*", ctx, matrix.getNRow(), matrix.getNCol());
	}

	public static RewriterStatement add(final RuleContext ctx, RewriterStatement... terms) {
		if (terms.length == 1)
			return terms[0];

		return new RewriterInstruction("+", ctx, new RewriterInstruction("argList", ctx, terms));
	}
}
