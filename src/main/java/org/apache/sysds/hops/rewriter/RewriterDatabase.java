package org.apache.sysds.hops.rewriter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

public class RewriterDatabase {

	private ConcurrentHashMap<RewriterStatementEntry, RewriterStatement> db = new ConcurrentHashMap<>();

	public void clear() {
		db.clear();
	}

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

	public void forEach(Consumer<RewriterStatement> consumer) {
		db.values().forEach(consumer);
	}

	public void parForEach(Consumer<RewriterStatement> consumer) {
		db.values().parallelStream().forEach(consumer);
	}

	public int size() {return db.size(); }

	public void serialize(BufferedWriter writer, final RuleContext ctx) throws IOException {
		for (RewriterStatement entry : db.values()) {
			writer.write("\n::STMT\n");
			writer.write(entry.toParsableString(ctx, true));
		}
	}

	public void deserialize(BufferedReader reader, final RuleContext ctx) throws IOException {
		List<String> strBuffer = new ArrayList<>();

		String line;
		while ((line = reader.readLine()) != null) {
			if (line.isBlank())
				continue;

			if (line.startsWith("::STMT")) {
				if (strBuffer.isEmpty())
					continue;
				try {
					RewriterStatement stmt = RewriterUtils.parse(String.join("\n", strBuffer), ctx);
					insertEntry(ctx, stmt);
					strBuffer.clear();
				} catch (Exception e) {
					System.err.println("An error occurred while parsing the string:\n" + String.join("\n", strBuffer));
					strBuffer.clear();
					e.printStackTrace();
				}
			} else {
				strBuffer.add(line);
			}
		}

		if (!strBuffer.isEmpty()) {
			try {
				RewriterStatement stmt = RewriterUtils.parse(String.join("\n", strBuffer), ctx);
				insertEntry(ctx, stmt);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
}
