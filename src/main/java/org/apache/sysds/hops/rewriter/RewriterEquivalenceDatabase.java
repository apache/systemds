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

package org.apache.sysds.hops.rewriter;

import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Consumer;

public class RewriterEquivalenceDatabase {
	private ConcurrentHashMap<RewriterStatementEntry, DBEntry> db = new ConcurrentHashMap<>();

	public void clear() {
		db.clear();
	}

	public boolean containsEntry(RewriterStatement instr) {
		return db.containsKey(instr);
	}

	public DBEntry insert(final RuleContext ctx, RewriterStatement canonicalForm, RewriterStatement equivalence) {
		return db.compute(new RewriterStatementEntry(ctx, canonicalForm), (k, v) -> {
			if (v == null)
				return new DBEntry(canonicalForm, equivalence);

			v.insertEquivalence(equivalence);
			return v;
		});
	}

	public DBEntry find(final RuleContext ctx, RewriterStatement canonicalForm) {
		return db.get(new RewriterStatementEntry(ctx, canonicalForm));
	}

	public void forEach(Consumer<DBEntry> consumer) {
		db.values().forEach(consumer);
	}

	public void parForEach(Consumer<DBEntry> consumer) {
		db.values().parallelStream().forEach(consumer);
	}

	public int size() {return db.size(); }

	@Deprecated
	public void serialize(BufferedWriter writer, final RuleContext ctx) throws IOException {
		for (DBEntry entry : db.values()) {
			writer.write("\n::STMT\n");
			writer.write(entry.canonicalForm.toParsableString(ctx, true));
		}
	}

	@Deprecated
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
					insert(ctx, stmt, null);
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
				insert(ctx, stmt, null);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	public static class DBEntry {
		public final RewriterStatement canonicalForm;
		public final List<RewriterStatement> equivalences;

		public DBEntry(RewriterStatement canonicalForm, RewriterStatement firstEquivalence) {
			this.canonicalForm = canonicalForm;
			this.equivalences = new ArrayList<>(3);

			if (firstEquivalence != null)
				this.equivalences.add(firstEquivalence);
		}

		public void insertEquivalence(RewriterStatement equivalence) {
			equivalences.add(equivalence);
		}
	}
}
