package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.mutable.MutableObject;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

public class RewriterAssertions {
	private final RuleContext ctx;
	private Map<RewriterStatement, RewriterAssertion> assertionMatcher = new HashMap<>();
	private Set<RewriterAssertion> allAssertions = new HashSet<>();

	public RewriterAssertions(final RuleContext ctx) {
		this.ctx = ctx;
	}

	/*public static RewriterAssertions ofExpression(RewriterStatement root, final RuleContext ctx) {
		Map<RewriterRule.IdentityRewriterStatement, RewriterAssertion> assertionMatcher = new HashMap<>();
		Set<RewriterAssertion> allAssertions = new HashSet<>();
		root.forEachPostOrder((cur, parent, pIdx) -> {
			if (cur.isInstruction() && cur.trueInstruction().equals("_EClass")) {
				Set<RewriterRule.IdentityRewriterStatement> mSet = cur.getChild(0).getOperands().stream().map(RewriterRule.IdentityRewriterStatement::new).collect(Collectors.toSet());
				RewriterAssertion newAssertion = RewriterAssertion.from(mSet);
				newAssertion.stmt = cur;
				allAssertions.add(newAssertion);
			}
		});

		RewriterAssertions assertions = new RewriterAssertions(ctx);
		assertions.allAssertions = allAssertions;
		assertions.assertionMatcher = assertionMatcher;

		root.unsafePutMeta("_assertions", assertions);
		return assertions;
	}*/

	public static RewriterAssertions copy(RewriterAssertions old, Map<RewriterStatement, RewriterStatement> createdObjects, boolean removeOthers) {
		//System.out.println("Copying: " + old);
		RewriterAssertions newAssertions = new RewriterAssertions(old.ctx);

		Map<RewriterAssertion, RewriterAssertion> mappedAssertions = new HashMap<>();

		newAssertions.allAssertions = old.allAssertions.stream().map(assertion -> {
			Set<RewriterStatement> newSet;

			if (removeOthers) {
				newSet = assertion.set.stream().map(el -> {
					RewriterStatement ret = createdObjects.get(el);
					//System.out.println("Found: " + el + " => " + ret);
					return ret;
				}).filter(Objects::nonNull).collect(Collectors.toSet());
				/*if (newSet.size() != assertion.set.size()) {
					System.out.println(createdObjects);
				}*/
			} else {
				newSet = assertion.set.stream().map(el -> createdObjects.getOrDefault(el, el)).collect(Collectors.toSet());
			}

			//System.out.println("NewSet: " + newSet);
			if (newSet.size() < 2)
				return null;

			RewriterAssertion mapped = RewriterAssertion.from(newSet);
			if (assertion.stmt != null)
				mapped.stmt = createdObjects.get(assertion.stmt);
			mappedAssertions.put(assertion, mapped);
			return mapped;
		}).filter(Objects::nonNull).collect(Collectors.toSet());

		if (removeOthers) {
			old.assertionMatcher.forEach((k, v) -> {
				RewriterStatement newK = createdObjects.get(k);

				if (newK == null)
					return;

				RewriterAssertion newV = mappedAssertions.get(v);

				if (newV == null)
					return;

				newAssertions.assertionMatcher.put(newK, newV);
			});
		} else {
			old.assertionMatcher.forEach((k, v) -> {
				RewriterStatement newK = createdObjects.getOrDefault(k, k);
				RewriterAssertion newV = mappedAssertions.get(v);

				if (newV == null)
					return;

				newAssertions.assertionMatcher.put(newK, newV);
			});
		}

		//System.out.println("New: " + newAssertions);

		return newAssertions;
	}

	/*public void update(Map<RewriterRule.IdentityRewriterStatement, RewriterRule.IdentityRewriterStatement> createdObjects) {
		for (RewriterAssertion assertion : allAssertions) {
			assertion.set = assertion.set.stream().map(el -> createdObjects.getOrDefault(el, el)).collect(Collectors.toSet());
			RewriterRule.IdentityRewriterStatement ids = new RewriterRule.IdentityRewriterStatement(assertion.stmt);
			assertion.stmt = createdObjects.getOrDefault(ids, ids).stmt;
		}

		Map<RewriterRule.IdentityRewriterStatement, RewriterAssertion> newAssertionMatcher = new HashMap<>();

		assertionMatcher.forEach((k, v) -> {
			newAssertionMatcher.put(createdObjects.getOrDefault(k, k), v);
		});

		assertionMatcher = newAssertionMatcher;
	}*/

	// TODO: What happens if the rewriter statement has already been instantiated? Updates will not occur
	public boolean addEqualityAssertion(RewriterStatement stmt1, RewriterStatement stmt2) {
		if (stmt1 == stmt2 || (stmt1.isLiteral() && stmt2.isLiteral() && stmt1.getLiteral().equals(stmt2.getLiteral())))
			return false;

		//if (!(stmt1 instanceof RewriterInstruction) || !(stmt2 instanceof RewriterInstruction))
		//	throw new UnsupportedOperationException("Asserting uninjectable objects is not yet supported: " + stmt1 + "; " + stmt2);

		//System.out.println("Asserting: " + stmt1 + " := " + stmt2);

		RewriterStatement e1 = stmt1;
		RewriterStatement e2 = stmt2;
		RewriterAssertion stmt1Assertions = assertionMatcher.get(e1);
		RewriterAssertion stmt2Assertions = assertionMatcher.get(e2);

		//System.out.println("Stmt1Assertion: " + stmt1Assertions);
		//System.out.println("Stmt2Assertion: " + stmt2Assertions);

		if (stmt1Assertions == stmt2Assertions) {
			if (stmt1Assertions == null) {
				// Then we need to introduce a new equality set
				Set<RewriterStatement> newSet = new HashSet<>();
				newSet.add(e1);
				newSet.add(e2);

				RewriterAssertion newAssertion = RewriterAssertion.from(newSet);

				assertionMatcher.put(e1, newAssertion);
				assertionMatcher.put(e2, newAssertion);

				allAssertions.add(newAssertion);

				resolveCyclicAssertions(newAssertion);
				//System.out.println("New assertion1: " + newAssertion);
				return true;
			}

			return false; // The assertion already exists
		}

		if (stmt1Assertions == null || stmt2Assertions == null) {
			boolean assert1 = stmt1Assertions == null;
			RewriterStatement toAssert = assert1 ? stmt1 : stmt2;
			RewriterAssertion existingAssertion = assert1 ? stmt2Assertions : stmt1Assertions;
			existingAssertion.set.add(toAssert);
			assertionMatcher.put(assert1 ? e1 : e2, existingAssertion);
			//System.out.println("Existing assertion: " + existingAssertion);
			if (existingAssertion.stmt != null)
				updateInstance(existingAssertion.stmt.getChild(0), existingAssertion.set);

			resolveCyclicAssertions(existingAssertion);
			//System.out.println("New assertion2: " + existingAssertion);
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
		if (stmt2Assertions.stmt != null)
			updateInstance(stmt2Assertions.stmt.getChild(0), stmt2Assertions.set);

		for (RewriterStatement stmt : stmt1Assertions.set)
			assertionMatcher.put(stmt, stmt2Assertions);

		if (stmt1Assertions.stmt != null)
			assertionMatcher.put(stmt1Assertions.stmt, stmt2Assertions); // Only temporary

		//System.out.println("New assertion3: " + stmt2Assertions);
		resolveCyclicAssertions(stmt2Assertions);

		return true;
	}

	// Replace cycles with _backRef()
	private void resolveCyclicAssertions(RewriterAssertion assertion) {
		if (assertion.stmt == null)
			return;

		//System.out.println("Resolving cycles in: " + assertion);

		String rType = assertion.stmt.getResultingDataType(ctx);

		RewriterStatement backref = new RewriterInstruction()
				.as(UUID.randomUUID().toString())
				.withInstruction("_backRef." + rType)
				.consolidate(ctx);
		backref.unsafePutMeta("_backRef", assertion.stmt);

		for (RewriterStatement eq : assertion.set) {
			eq.forEachPreOrder((cur, parent, pIdx) -> {
				for (int i = 0; i < cur.getOperands().size(); i++)
					if (getAssertionObj(cur.getChild(i)) == assertion)
						cur.getOperands().set(i, backref);

				return true;
			});
		}
	}

	public RewriterAssertion getAssertionObj(RewriterStatement stmt) {
		return assertionMatcher.get(stmt);
	}

	public Set<RewriterStatement> getAssertions(RewriterStatement stmt) {
		RewriterAssertion set = assertionMatcher.get(stmt);
		return set == null ? Collections.emptySet() : set.set;
	}

	public RewriterStatement getAssertionStatement(RewriterStatement stmt, RewriterStatement parent) {
		//System.out.println("Checking: " + stmt);
		//System.out.println("In: " + this);
		RewriterAssertion set = assertionMatcher.get(stmt);

		if (set == null)
			return stmt;

		RewriterStatement mstmt = set.stmt;

		if (mstmt == null) {
			// Then we create a new statement for it
			RewriterStatement argList = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("argList").withOps(set.set.toArray(RewriterStatement[]::new));
			mstmt = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("_EClass").withOps(argList);
			mstmt.consolidate(ctx);
			set.stmt = mstmt;
			assertionMatcher.put(set.stmt, set);
			resolveCyclicAssertions(set);
		} else if (mstmt.getChild(0) == parent) {
			return stmt;
		}

		return mstmt;
	}

	// TODO: This does not handle metadata
	public RewriterStatement update(RewriterStatement root) {
		RewriterStatement eClass = getAssertionStatement(root, null);

		if (eClass == null)
			eClass = root;
		else if (root.getMeta("_assertions") != null)
			eClass.unsafePutMeta("_assertions", root.getMeta("_assertions"));

		updateRecursively(eClass);

		return eClass;
	}

	private void updateRecursively(RewriterStatement cur) {
		for (int i = 0; i < cur.getOperands().size(); i++) {
			RewriterStatement child = cur.getChild(i);
			RewriterStatement eClass = getAssertionStatement(child, cur);

			if (eClass != child)
				cur.getOperands().set(i, eClass);

			updateRecursively(cur.getChild(i));
		}
	}

	// TODO: We have to copy the assertions to the root node if it changes
	/*public RewriterStatement buildEquivalences(RewriterStatement stmt) {
		RewriterStatement mAssert = getAssertionStatement(stmt);

		mAssert.forEachPreOrder((cur, parent, pIdx) -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement op = cur.getOperands().get(i);
				RewriterStatement asserted = getAssertionStatement(op);

				if (asserted != op && asserted.getOperands().get(0) != cur)
					cur.getOperands().set(i, asserted);
			}

			return true;
		});

		mAssert.prepareForHashing();
		mAssert.recomputeHashCodes(ctx);

		return mAssert;
	}*/

	@Override
	public String toString() {
		return allAssertions.toString();
	}

	private void updateInstance(RewriterStatement stmt, Set<RewriterStatement> set) {
		if (stmt != null) {
			stmt.getOperands().clear();
			stmt.getOperands().addAll(set);
		}
	}

	private static class RewriterAssertion {
		Set<RewriterStatement> set;
		RewriterStatement stmt;

		@Override
		public String toString() {
			//throw new IllegalArgumentException();
			if (stmt != null)
				return stmt.toString() + " -- " + System.identityHashCode(this);

			return set.toString() + " -- " + System.identityHashCode(this);
		}

		static RewriterAssertion from(Set<RewriterStatement> set) {
			RewriterAssertion a = new RewriterAssertion();
			a.set = set;
			return a;
		}
	}
}
