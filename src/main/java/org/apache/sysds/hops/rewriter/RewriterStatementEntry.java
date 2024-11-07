package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;

import java.util.HashMap;

public class RewriterStatementEntry {
	private final RuleContext ctx;
	final RewriterStatement instr;

	public RewriterStatementEntry(final RuleContext ctx, RewriterStatement instr) {
		this.ctx = ctx;
		this.instr = instr;
	}

	@Override
	public int hashCode() {
		return instr.structuralHashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (o instanceof RewriterStatement) {
			if (instr == o)
				return true;
			if (instr.structuralHashCode() != ((RewriterStatement)o).structuralHashCode())
				return false;
			return instr.match(new RewriterStatement.MatcherContext(ctx, (RewriterStatement) o, null, -1, (RewriterStatement) o, false, false, false, false, false, false, true, false, false, new HashMap<>()));
		}

		if (o.hashCode() != hashCode())
			return false;

		if (o instanceof RewriterStatementEntry) {
			if (instr == ((RewriterStatementEntry) o).instr)
				return true;
			return instr.match(new RewriterStatement.MatcherContext(ctx, ((RewriterStatementEntry) o).instr, null, -1, ((RewriterStatementEntry) o).instr, false, false, false, false, false, false, true, false, false, new HashMap<>()));
		}
		return false;
	}
}
