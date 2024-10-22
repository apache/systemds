package org.apache.sysds.hops.rewriter;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

public class RewriterAssertions {
	private final RuleContext ctx;
	private Map<RewriterRule.IdentityRewriterStatement, RewriterAssertion> assertionMatcher = new HashMap<>();
	private Set<RewriterAssertion> allAssertions = new HashSet<>();

	public RewriterAssertions(final RuleContext ctx) {
		this.ctx = ctx;
	}

	// TODO: What happens if the rewriter statement has already been instantiated? Updates will not occur
	public boolean addEqualityAssertion(RewriterStatement stmt1, RewriterStatement stmt2) {
		if (!(stmt1 instanceof RewriterInstruction) || !(stmt2 instanceof RewriterInstruction))
			throw new UnsupportedOperationException("Asserting uninjectable objects is not yet supported");

		if (stmt1 == stmt2)
			return false;

		RewriterRule.IdentityRewriterStatement e1 = new RewriterRule.IdentityRewriterStatement(stmt1);
		RewriterRule.IdentityRewriterStatement e2 = new RewriterRule.IdentityRewriterStatement(stmt2);
		RewriterAssertion stmt1Assertions = assertionMatcher.get(e1);
		RewriterAssertion stmt2Assertions = assertionMatcher.get(e2);

		if (stmt1Assertions == stmt2Assertions) {
			if (stmt1Assertions == null) {
				// Then we need to introduce a new equality set
				Set<RewriterRule.IdentityRewriterStatement> newSet = new HashSet<>();
				newSet.add(e1);
				newSet.add(e2);

				RewriterAssertion newAssertion = RewriterAssertion.from(newSet);

				assertionMatcher.put(e1, newAssertion);
				assertionMatcher.put(e2, newAssertion);

				allAssertions.add(newAssertion);

				return true;
			}

			return false; // The assertion already exists
		}

		if (stmt1Assertions == null || stmt2Assertions == null) {
			boolean assert1 = stmt1Assertions == null;
			RewriterRule.IdentityRewriterStatement toAssert = new RewriterRule.IdentityRewriterStatement(assert1 ? stmt1 : stmt2);
			RewriterAssertion existingAssertion = assert1 ? stmt2Assertions : stmt1Assertions;
			existingAssertion.set.add(toAssert);
			assertionMatcher.put(assert1 ? e1 : e2, existingAssertion);
			updateInstance(existingAssertion.stmt, existingAssertion.set);
			return true;
		}

		// Otherwise we need to merge the assertions

		// For that, we choose the smaller set as we will need fewer operations
		if (stmt1Assertions.set.size() > stmt2Assertions.set.size()) {
			RewriterAssertion tmp = stmt1Assertions;
			stmt1Assertions = stmt2Assertions;
			stmt2Assertions = tmp;
		}

		stmt2Assertions.set.addAll(stmt1Assertions.set);
		allAssertions.remove(stmt1Assertions);
		updateInstance(stmt2Assertions.stmt, stmt2Assertions.set);

		for (RewriterRule.IdentityRewriterStatement stmt : stmt1Assertions.set)
			assertionMatcher.put(stmt, stmt2Assertions);

		return true;
	}

	public Set<RewriterRule.IdentityRewriterStatement> getAssertions(RewriterStatement stmt) {
		RewriterAssertion set = assertionMatcher.get(new RewriterRule.IdentityRewriterStatement(stmt));
		return set == null ? Collections.emptySet() : set.set;
	}

	public RewriterStatement getAssertionStatement(RewriterStatement stmt) {
		RewriterAssertion set = assertionMatcher.get(new RewriterRule.IdentityRewriterStatement(stmt));

		if (set == null)
			return stmt;

		RewriterStatement mstmt = set.stmt;

		if (mstmt == null) {
			// Then we create a new statement for it
			mstmt = new RewriterInstruction().as(UUID.randomUUID().toString()).ofType(stmt.getResultingDataType(ctx)).withInstruction("_EClass").withOps(set.set.stream().map(id -> id.stmt).toArray(RewriterStatement[]::new));
			mstmt.consolidate(ctx);
			set.stmt = mstmt;
		}

		return mstmt;
	}

	private void updateInstance(RewriterStatement stmt, Set<RewriterRule.IdentityRewriterStatement> set) {
		if (stmt != null) {
			stmt.getOperands().clear();
			stmt.getOperands().addAll(set.stream().map(id -> id.stmt).collect(Collectors.toList()));
		}
	}

	/*public void applyAssertions(RewriterStatement root) {
		Map<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement> repl = new HashMap<>();



		root.forEachPostOrder((cur, parent, pIdx) -> {
			Set<RewriterRule.IdentityRewriterStatement> set = getAssertions(cur);

			if (set.isEmpty())
				return;

			// For now, we assume that all assertions are of type RewriterInstruction
			RewriterInstruction instr = (RewriterInstruction) cur;
			RewriterInstruction cpy = (RewriterInstruction) instr.copyNode();

			// Now, we use the old object as container for my equivalence relation
			instr.unsafeSetInstructionName("eqSet");
			instr.getOperands().clear();
			instr.getOperands().addAll(set.stream().map(el -> el.stmt).collect(Collectors.toList()));
		});

		root.prepareForHashing();
		root.recomputeHashCodes(ctx);
	}*/

	private static class RewriterAssertion {
		Set<RewriterRule.IdentityRewriterStatement> set;
		RewriterStatement stmt;

		static RewriterAssertion from(Set<RewriterRule.IdentityRewriterStatement> set) {
			RewriterAssertion a = new RewriterAssertion();
			a.set = set;
			return a;
		}
	}
}
