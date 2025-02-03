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

import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiFunction;

// We assume that _argList() will have one unique parent
public class TopologicalSort {
	protected static final Log LOG = LogFactory.getLog(TopologicalSort.class.getName());

	public static boolean DEBUG = false;

	// All of these operators are sortable with argument lists (e.g. +(argList(1, 2, 3))
	private static final Set<String> SORTABLE_ARGLIST_OPS = Set.of("+", "*", "_idxExpr", "_EClass", "rand", "_dummy");
	// All of these operators are sortable but have their operands directly as children (e.g. ==(a,b))
	private static final Set<String> SORTABLE_OPS = Set.of("==", "!=");

	public static void sort(RewriterStatement root, final RuleContext ctx) {
		sort(root, (el, parent) -> {
			if (!el.isInstruction())
				return false;

			if (el.isArgumentList())
				return parent != null && SORTABLE_ARGLIST_OPS.contains(parent.trueInstruction());

			return SORTABLE_OPS.contains(el.trueInstruction());
		}, ctx);
	}

	public static void sort(RewriterStatement root, BiFunction<RewriterStatement, RewriterStatement, Boolean> isArrangable, final RuleContext ctx) {
		// First, we setup an artificial root node to be able to sort E-Classes that are only included as meta-info not directly in the operand structure
		Set<RewriterStatement> hiddenEClasses = new HashSet<>();
		root.forEachPostOrder((stmt, pred) -> {
			if (stmt instanceof RewriterDataType && !stmt.isLiteral() && stmt.getResultingDataType(ctx).equals("MATRIX")) {
				if (stmt.getNRow().isInstruction() && stmt.getNRow().trueInstruction().equals("_EClass"))
					hiddenEClasses.add(stmt.getNRow());

				if (stmt.getNCol().isInstruction() && stmt.getNCol().trueInstruction().equals("_EClass"))
					hiddenEClasses.add(stmt.getNCol());
			}
		}, true);

		RewriterStatement oldRoot = root;

		if (!hiddenEClasses.isEmpty()) {
			RewriterStatement argList = new RewriterInstruction().withInstruction("argList").withOps(hiddenEClasses.toArray(RewriterStatement[]::new));
			RewriterStatement dummy = new RewriterInstruction().withInstruction("_dummy").withOps(argList);
			root = new RewriterInstruction().withInstruction("_root").withOps(root, dummy);
		}

		List<RewriterStatement> uncertainParents = setupOrderFacts(root, isArrangable, ctx);

		buildAddresses(root, ctx);
		resolveAmbiguities(root, ctx, uncertainParents);
		resetAddresses(uncertainParents);

		int factCtr = 0;

		// Now, we start introducing facts for the lowest level unordered sets
		Set<UnorderedSet> lowestUncertainties = findLowestUncertainties(root);
		int ctr = 0;

		while (!lowestUncertainties.isEmpty()) {
			if (DEBUG) {
				LOG.trace("Uncertainties after iteration " + ctr + ": " + lowestUncertainties.size());
				LOG.trace("Lowest uncertainties: " + lowestUncertainties);
			}

			factCtr = introduceFacts(lowestUncertainties, factCtr);
			buildAddresses(root, ctx);

			if (DEBUG) {
				LOG.trace("Built addresses:");
				for (UnorderedSet u : lowestUncertainties) {
					for (RewriterStatement s : u.contents) {
						LOG.trace("- " + s + " :: " + getAddress(s));
					}
				}
			}

			resolveAmbiguities(root, ctx, uncertainParents);
			resetAddresses(uncertainParents);

			lowestUncertainties = findLowestUncertainties(root);
			ctr++;

			if (ctr > 100)
				throw new RuntimeException("Could not finish sorting process for expression:\n" + root.toParsableString(ctx)); // Should never get here but just to make sure
		}

		// At the end
		if (DEBUG)
			LOG.trace("Before construction: " + oldRoot.toParsableString(ctx));
		constructNewDAG(oldRoot, ctx);
		if (DEBUG)
			LOG.trace("After construction: " + oldRoot.toParsableString(ctx));
	}

	// Returns all uncertain parents ordered in post order (elements without uncertain sub-DAGs come first in the list)
	private static List<RewriterStatement> setupOrderFacts(RewriterStatement root, BiFunction<RewriterStatement, RewriterStatement, Boolean> isArrangable, final RuleContext ctx) {
		List<RewriterStatement> uncertainParents = new ArrayList<>();

		// Create a random global order which will be used for indistinguishable sub-DAGs
		MutableInt nameCtr = new MutableInt(0);
		root.forEachPostOrder((el, pred) -> {
			if (el.isLiteral())
				return;

			el.unsafePutMeta("_tempName", nameCtr.intValue());
			nameCtr.increment();
			boolean arrangable = isArrangable.apply(el, pred.getParent());

			el.unsafePutMeta("_arrangable", arrangable);
		}, false);

		// Try to establish a first order
		root.forEachPostOrder((el, pred) -> {
			if (el.isLiteral())
				return;

			boolean arrangable = (boolean) el.getMeta("_arrangable");

			List<Object> knownOrder = new ArrayList<>();
			el.unsafePutMeta("_knownOrder", knownOrder);

			if (arrangable) {
				el.getOperands().sort((cmp1, cmp2) -> compare(cmp1, cmp2, ctx));

				boolean containsUnorderedSet = false;

				List<RewriterStatement> currSet = new ArrayList<>();
				currSet.add(el.getOperands().get(0));

				for (int i = 1; i < el.getOperands().size(); i++) {
					if (compare(el.getOperands().get(i-1), el.getOperands().get(i), ctx) != 0) {
						if (currSet.size() == 1) {
							knownOrder.add(currSet.get(0));
							currSet.clear();
						} else {
							final RewriterStatement first = currSet.get(0);
							if (currSet.stream().allMatch(mEl -> first == mEl)) {
								// Then this is not an unordered set as it only contains one instance and the order doesn't matter
								knownOrder.addAll(currSet);
								currSet.clear();
							} else {
								containsUnorderedSet = true;
								currSet.forEach(cur -> {
									if (!cur.isLiteral())
										cur.unsafePutMeta("_addresses", new ArrayList<String>());
								});
								knownOrder.add(new UnorderedSet(currSet));
								currSet = new ArrayList<>();
							}
						}
					}

					currSet.add(el.getOperands().get(i));
				}

				if (currSet.size() == 1)
					knownOrder.add(currSet.get(0));
				else {
					final RewriterStatement first = currSet.get(0);
					if (currSet.stream().allMatch(first::equals)) {
						knownOrder.addAll(currSet);
					} else {
						containsUnorderedSet = true;
						currSet.forEach(cur -> {
							if (!cur.isLiteral())
								cur.unsafePutMeta("_addresses", new ArrayList<String>());
						});
						knownOrder.add(new UnorderedSet(currSet));
					}
				}

				if (containsUnorderedSet)
					uncertainParents.add(el);
			} else {
				knownOrder.addAll(el.getOperands());
			}

			if (DEBUG)
				LOG.trace("Initial known order of " + el.toParsableString(ctx) + ": " + knownOrder);
		}, false);

		return uncertainParents;
	}

	private static int introduceFacts(Collection<UnorderedSet> sets, int factCtr) {
		for (RewriterStatement stmt : allChildren(sets)) {
			if (stmt.isLiteral())
				continue;

			if (stmt.getMeta("_addresses") == null)
				stmt.unsafePutMeta("_addresses", new ArrayList<>());

			if (stmt.getMeta("_fact") == null)
				stmt.unsafePutMeta("_fact", factCtr++);
		}

		return factCtr;
	}

	// Returns a list of all unordered set that do not contain other unordered set
	private static Set<UnorderedSet> findLowestUncertainties(RewriterStatement root) {
		Set<UnorderedSet> set = new HashSet<>();
		recursivelyFindLowestUncertainties(root, set);

		List<UnorderedSet> tmpList = new ArrayList<>(set);
		Set<UnorderedSet> minSet = new HashSet<>();
		// We have the issue that uncertainties might still depend on each other (e.g. {a,b}, {inv(a),inv(b)}), even if they are the lowest entries
		// Theoretically, this comparison might still lead to amgibuities, but never occurred in our examples
		int minCumSize = Integer.MAX_VALUE;
		for (int i = 0; i < tmpList.size(); i++) {
			int cumSize = tmpList.get(i).contents.stream().map(RewriterStatement::countInstructions).reduce(0, Integer::sum);

			if (cumSize < minCumSize) {
				minSet.clear();
				minCumSize = cumSize;
			}

			if (cumSize <= minCumSize)
				minSet.add(tmpList.get(i));
		}

		return minSet;
	}

	// All children in post order and unique
	private static List<RewriterStatement> allChildren(Collection<UnorderedSet> unorderedSets) {
		Set<RewriterStatement> is = new HashSet<>();
		List<RewriterStatement> children = new ArrayList<>();
		for (UnorderedSet set : unorderedSets)
			for (RewriterStatement s : set.contents)
				traverse(s, is, children);

		return children;
	}

	private static void traverse(RewriterStatement stmt, Set<RewriterStatement> visited, List<RewriterStatement> l) {
		if (visited.contains(stmt))
			return;

		visited.add(stmt);
		stmt.getOperands().forEach(el -> traverse(el, visited, l));

		l.add(stmt);
	}

	private static boolean recursivelyFindLowestUncertainties(RewriterStatement current, Set<UnorderedSet> lowestUncertainties) {
		if (current.isLiteral())
			return false;

		List<Object> knownOrder = (List<Object>) current.getMeta("_knownOrder");
		boolean containsUncertainty = false;

		for (Object o : knownOrder) {
			if (o instanceof RewriterStatement) {
				containsUncertainty |= recursivelyFindLowestUncertainties((RewriterStatement) o, lowestUncertainties);
			} else {
				UnorderedSet set = (UnorderedSet) o;
				containsUncertainty = true;
				boolean foundEmbeddedUncertainty = set.contents.stream().anyMatch(stmt -> recursivelyFindLowestUncertainties(stmt, lowestUncertainties));

				if (foundEmbeddedUncertainty)
					lowestUncertainties.remove(set);
				else
					lowestUncertainties.add(set);
			}
		}

		return containsUncertainty;
	}

	public static void constructNewDAG(RewriterStatement root, final RuleContext ctx) {
		root.forEachPostOrder((cur, pred) -> {
			List<Object> knownOrder = (List<Object>) cur.getMeta("_knownOrder");
			if (DEBUG)
				LOG.trace("KnownOrder of " + cur.toParsableString(ctx) + ": " + knownOrder);

			for (int i = 0; i < cur.getOperands().size(); i++)
				cur.getOperands().set(i, (RewriterStatement) knownOrder.get(i));

			cur.unsafeRemoveMeta("_knownOrder");
			cur.unsafeRemoveMeta("_addresses");
			cur.unsafeRemoveMeta("_address");
			cur.unsafeRemoveMeta("_arrangable");
			cur.unsafeRemoveMeta("_tempName");
		}, false);

		root.prepareForHashing();
		root.recomputeHashCodes(ctx);
	}

	// Here, we try to infer new information given the address information
	// This step also resets all addresses as they will change after one sorting step
	private static boolean resolveAmbiguities(RewriterStatement root, final RuleContext ctx, List<RewriterStatement> uncertainParents) {
		boolean couldResolve = false;
		boolean couldResolveAnyUncertainty = true;

		while (couldResolveAnyUncertainty) {
			couldResolveAnyUncertainty = false;

			for (int i = 0; i < uncertainParents.size(); i++) {
				List<Object> knownOrder = (List<Object>) uncertainParents.get(i).getMeta("_knownOrder");
				boolean uncertaintyRemaining = false;

				for (int j = 0; j < knownOrder.size(); j++) {
					if (knownOrder.get(j) instanceof UnorderedSet) {
						UnorderedSet set = (UnorderedSet) knownOrder.get(j);

						if (tryResolveUncertainties(set, ctx)) {
							couldResolveAnyUncertainty = true;
							couldResolve = true;
							knownOrder.set(j, set.contents.get(0));
							knownOrder.addAll(j+1, set.contents.subList(1, set.contents.size()));
							set.contents.forEach(el -> {
								el.unsafeRemoveMeta("_addresses");
								el.unsafeRemoveMeta("_address");
							});
						} else {
							uncertaintyRemaining = true;
						}
					}
				}

				if (!uncertaintyRemaining) {
					uncertainParents.remove(i);
					i--;
				}
			}
		}

		return couldResolve;
	}

	private static void resetAddresses(List<RewriterStatement> uncertainParents) {
		for (RewriterStatement uParent : uncertainParents) {
			List<Object> knownOrder = (List<Object>) uParent.getMeta("_knownOrder");

			for (Object o : knownOrder) {
				if (o instanceof UnorderedSet) {
					((UnorderedSet) o).contents.forEach(el -> {
						List<String> addresses = (List<String>) el.getMeta("_addresses");

						if (addresses == null) {
							addresses = new ArrayList<>();
							el.unsafePutMeta("_addresses", addresses);
							el.unsafeRemoveMeta("_address");
						}

						addresses.clear();
					});
				}
			}
		}
	}

	private static boolean tryResolveUncertainties(UnorderedSet set, final RuleContext ctx) {
		set.contents.sort((el1, el2) -> compare(el1, el2, ctx)); // We assume that every statement has an address, as it is uncertain

		RewriterStatement compareTo = set.contents.get(0);
		// Check if ambiguity could be resolved
		for (int i = 1; i < set.contents.size(); i++) {
			if (compareTo.equals(set.contents.get(i)))
				continue; // Ignore same instances

			if (compare(set.contents.get(i), compareTo, ctx) == 0)
				return false; // Then there are still some ambiguities

			compareTo = set.contents.get(i);
		}

		return true;
	}

	private static List<RewriterStatement> buildAddresses(RewriterStatement root, final RuleContext ctx) {
		// First, catch all addresses
		List<RewriterStatement> elementsWithAddress = new ArrayList<>();
		recursivelyBuildAddresses(root, null, ctx, elementsWithAddress);

		// Now, we sort all addresses
		for (RewriterStatement el : elementsWithAddress) {
			List<String> addresses = (List<String>) el.getMeta("_addresses");
			Collections.sort(addresses);
			String address = String.join(";", addresses);
			el.unsafePutMeta("_address", address);

			if (DEBUG)
				LOG.trace("Address of " + el + " :: " + address);
		}

		return elementsWithAddress;
	}

	private static void recursivelyBuildAddresses(RewriterStatement current, String currentAddress, final RuleContext ctx, List<RewriterStatement> elementsWithAddress) {
		List<Object> knownOrder = (List<Object>)current.getMeta("_knownOrder");
		List<String> addresses = (List<String>)current.getMeta("_addresses");

		if (knownOrder == null)
			knownOrder = Collections.emptyList();



		if (DEBUG) {
			LOG.trace("CUR: " + current);
			LOG.trace("KnownOrder: " + knownOrder);
		}

		if (addresses != null) {
			if (addresses.isEmpty())
				elementsWithAddress.add(current);

			addresses.add(currentAddress);
		}

		for (int i = 0; i < knownOrder.size(); i++) {
			Object next = knownOrder.get(i);
			String addr = currentAddress == null ? Integer.toString(i) : currentAddress + "." + i;

			if (next instanceof RewriterStatement) {
				recursivelyBuildAddresses((RewriterStatement) next, addr, ctx, elementsWithAddress);
			} else {
				UnorderedSet set = (UnorderedSet) next;
				set.contents.forEach(el -> recursivelyBuildAddresses(el, addr, ctx, elementsWithAddress));
			}
		}
	}

	private static String getAddress(RewriterStatement stmt) {
		String addr = (String) stmt.getMeta("_address");

		if (addr == null)
			return null;

		return addr + (stmt.getMeta("_fact") == null ? "_" : "_" + stmt.getMeta("_fact"));
	}

	// Expects that the children have already been sorted to the best of the current knowledge
	public static int compare(RewriterStatement stmt1, RewriterStatement stmt2, final RuleContext ctx) {
		int comp = toOrderString(ctx, stmt1, false).compareTo(toOrderString(ctx, stmt2, false));

		if (comp != 0 || stmt1.equals(stmt2))
			return comp;

		List<Object> knownOrder1 = (List<Object>)stmt1.getMeta("_knownOrder");
		List<Object> knownOrder2 = (List<Object>)stmt2.getMeta("_knownOrder");

		// Then the two statements are distinguishable by their number of unknowns
		if (knownOrder1.size() != knownOrder2.size())
			return Integer.compare(knownOrder1.size(), knownOrder2.size());

		for (int i = 0; i < knownOrder1.size() && comp == 0; i++)
			comp = compare(knownOrder1.get(i), knownOrder2.get(i), ctx);

		if (comp == 0) {
			String addr1 = getAddress(stmt1);
			String addr2 = getAddress(stmt2);

			if (addr1 != null && addr2 != null)
				return addr1.compareTo(addr2);
		}

		return comp;
	}

	public static int compare(Object obj1, Object obj2, final RuleContext ctx) {
		boolean isStmt1 = obj1 instanceof RewriterStatement;
		boolean isStmt2 = obj2 instanceof RewriterStatement;

		if (isStmt1 && !isStmt2)
			return 1;
		if (!isStmt1 && isStmt2)
			return -1;

		if (isStmt1 && isStmt2)
			return compare((RewriterStatement) obj1, (RewriterStatement) obj2, ctx);

		UnorderedSet set1 = (UnorderedSet) obj1;
		UnorderedSet set2 = (UnorderedSet) obj2;

		if (set1.contents.size() < 2 || set2.contents.size() < 2)
			throw new IllegalArgumentException(); // This should never happen as this would not be an unknown ordering

		if (set1.contents.size() != set2.contents.size())
			return Integer.compare(set1.contents.size(), set2.contents.size());

		// Now, we can just choose any representant of the set
		return compare(set1.contents.get(0), set2.contents.get(0), ctx);
	}

	public static String toOrderString(final RuleContext ctx, RewriterStatement stmt, boolean useGlobalOrder) {
		String globalOrderAddition = useGlobalOrder ? ((Integer)stmt.getMeta("_tempName")).toString() : "";

		if (stmt.isInstruction()) {
			return stmt.getResultingDataType(ctx) + ":" + stmt.trueTypedInstruction(ctx) + "[" + stmt.refCtr + "](" + stmt.getOperands().size() + ")" + globalOrderAddition + ";";
		} else {
			return stmt.getResultingDataType(ctx) + ":" + (stmt.isLiteral() ? "L:" + stmt.getLiteral() : "V") + "[" + stmt.refCtr + "](0)" + globalOrderAddition + ";";
		}
	}



	static class UnorderedSet {
		List<RewriterStatement> contents;

		public UnorderedSet(List<RewriterStatement> contents) {
			this.contents = contents;
		}

		public String toString() {
			return contents.toString();
		}
	}
}