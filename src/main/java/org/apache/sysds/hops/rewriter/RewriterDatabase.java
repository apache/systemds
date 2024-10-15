package org.apache.sysds.hops.rewriter;

import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.ConcurrentHashMap;

public class RewriterDatabase {

	private ConcurrentHashMap<RewriterStatementEntry, RewriterStatement> db = new ConcurrentHashMap<>();

	public boolean containsEntry(RewriterStatement instr) {
		return db.containsKey(instr);
	}

	public boolean insertEntry(final RuleContext ctx, RewriterStatement stmt) {
		return db.putIfAbsent(new RewriterStatementEntry(ctx, stmt), stmt) == null;
	}

	public RewriterStatement find(final RuleContext ctx, RewriterStatement stmt) {
		return db.get(new RewriterStatementEntry(ctx, stmt));
	}

	public RewriterStatement insertOrReturn(final RuleContext ctx, RewriterStatement stmt) {
		return db.putIfAbsent(new RewriterStatementEntry(ctx, stmt), stmt);
	}

	public int size() {return db.size(); }
}
