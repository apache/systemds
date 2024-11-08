package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import scala.App;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class RewriterRuleCreator {

	private RuleContext ctx;
	private RewriterRuleSet ruleSet;
	private List<RewriterRule> activeRules;

	public RewriterRuleCreator(final RuleContext ctx) {
		this.ctx = ctx;
		activeRules = Collections.synchronizedList(new LinkedList<>());
		ruleSet = new RewriterRuleSet(ctx, activeRules);
	}

	public synchronized void forEachRule(Consumer<RewriterRule> consumer) {
		activeRules.forEach(consumer);
	}

	public synchronized boolean registerRule(RewriterRule rule, long preCost, long postCost) {
		// First, we check if an existing rule already applies an equivalent rewrite (cost wise)
		RewriterStatement toTest = rule.getStmt1().nestedCopy(false);

		boolean converged = false;
		boolean changed = false;

		for (int i = 0; i < 1000; i++) {
			RewriterRuleSet.ApplicableRule applicableRule = ruleSet.acceleratedFindFirst(toTest);

			if (applicableRule == null) {
				converged = true;
				break; // Then we converged
			}

			toTest = applicableRule.rule.apply(applicableRule.matches.get(0), toTest, applicableRule.forward, false);
			changed = true;
		}

		if (!converged)
			throw new IllegalArgumentException("The existing rule-set did not seem to converge for the example: \n" + toTest.toParsableString(ctx, true));

		if (changed) {
			long existingPostCost;

			try {
				existingPostCost = RewriterCostEstimator.estimateCost(toTest, el -> 2000L, ctx);
			} catch (Exception e) {
				System.err.println("Err in cost from orig: " + rule.getStmt1().toParsableString(ctx));
				e.printStackTrace();
				return false;
			}

			if (existingPostCost <= postCost)
				return false; // Then this rule is not beneficial
		}

		RewriterRuleSet probingSet = new RewriterRuleSet(ctx, List.of(rule));
		List<RewriterRule> rulesToRemove = new ArrayList<>();
		List<RewriterRule> rulesThatMustComeBefore = new ArrayList<>();

		// Check for interactions between different rules
		for (RewriterRule existingRule : activeRules) {
			RewriterStatement mProbe = existingRule.getStmt1();
			RewriterRuleSet.ApplicableRule applicableRule = probingSet.acceleratedFindFirst(mProbe);

			if (applicableRule != null) {
				// Then we have to take a deeper look into the interaction between the rules
				// Either the new rule achieves a better result -> the old rule can be eliminated
				// Or the new rule finds a worse rewrite for the existing rule -> Then the existing rule must be kept and be applied before the new rule
				mProbe = mProbe.nestedCopy(true);

				for (int i = 0; i < 20; i++) {
					applicableRule = probingSet.acceleratedFindFirst(mProbe);

					if (i == 19)
						throw new IllegalArgumentException("The following rule created a conflict with another rule:\nNew one:\n" + rule + "\nExisting:\n" + existingRule);
					if (applicableRule != null)
						mProbe = applicableRule.rule.apply(applicableRule.matches.get(0), mProbe, applicableRule.forward, false);
					else
						break;
				}

				long newCost = mProbe.getCost(ctx);
				long existingRuleNewCost = existingRule.getStmt2().getCost(ctx);

				if (newCost == -1 || existingRuleNewCost == -1)
					throw new IllegalArgumentException("The rule set or the new rule resulted in an invalid cost:\nNew one:\n" + rule + "\nExisting:\n" + existingRule);

				if (newCost <= existingRuleNewCost) {
					// Then we remove the old rule
					rulesToRemove.add(existingRule);
				} else {
					// Then the existing rule is still legitimate and must come before the new rule as it is more specific
					rulesThatMustComeBefore.add(existingRule);
				}
			}
		}

		activeRules.removeAll(rulesToRemove);

		// Now, we include the rule to the system
		// TODO: Check, if we can eliminate an existing rule instead as the new one is more general
		// TODO: Further checks are needed, especially if the new heuristic converges in all cases
		activeRules.add(rule);

		ruleSet.accelerate();
		return true;
	}






	///// STATIC METHODS /////

	public static RewriterRule createRule(RewriterStatement from, RewriterStatement to, RewriterStatement canonicalForm1, RewriterStatement canonicalForm2, final RuleContext ctx) {
		from = from.nestedCopy(true);
		to = to.nestedCopy(true);
		Map<RewriterStatement, RewriterStatement> assocs = getAssociations(from, to, canonicalForm1, canonicalForm2, ctx);

		// Now, we replace all variables with a common element
		from.forEachPreOrder((cur, parent, pIdx) -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement child = cur.getChild(i);

				if (child instanceof RewriterDataType && !child.isLiteral()) {
					RewriterStatement newRef = assocs.get(child);

					//if (newRef == null)
					//	throw new IllegalArgumentException("Null assoc for: " + child + "\nIn:\n" + fFrom.toParsableString(ctx) + "\n" + fTo.toParsableString(ctx) + "\n" + canonicalForm1.toParsableString(ctx));

					if (newRef != null)
						cur.getOperands().set(i, newRef);
				}
			}

			return true;
		});

		RewriterRule rule = new RewriterRuleBuilder(ctx, "Autogenerated rule").setUnidirectional(true).completeRule(from, to).build();
		return rule;
	}

	private static Map<RewriterStatement, RewriterStatement> getAssociations(RewriterStatement from, RewriterStatement to, RewriterStatement canonicalFormFrom, RewriterStatement canonicalFormTo, final RuleContext ctx) {
		Map<RewriterStatement, RewriterStatement> fromCanonicalLink = getAssociationToCanonicalForm(from, canonicalFormFrom, true);
		Map<RewriterStatement, RewriterStatement> toCanonicalLink = getAssociationToCanonicalForm(to, canonicalFormTo, true);

		RewriterStatement.MatcherContext matcher = RewriterStatement.MatcherContext.exactMatch(ctx, canonicalFormTo);
		canonicalFormFrom.match(matcher);

		Map<RewriterStatement, RewriterStatement> assocs = new HashMap<>();
		matcher.getDependencyMap().forEach((k, v) -> {
			if (k.isLiteral())
				return;

			RewriterStatement newKey = fromCanonicalLink.get(k);
			RewriterStatement newValue = toCanonicalLink.get(v);

			if (newKey == null || newValue == null)
				throw new IllegalArgumentException("Null reference detected!");

			assocs.put(newKey, newValue);
		});

		return assocs;
	}

	private static Map<RewriterStatement, RewriterStatement> getAssociationToCanonicalForm(RewriterStatement stmt, RewriterStatement canonicalForm, boolean reversed) {
		// We identify all associations by their names
		// If there are name collisions, this does not work
		Map<String, RewriterStatement> namedVariables = new HashMap<>();
		stmt.forEachPostOrder((cur, parent, pIdx) -> {
			if (!(cur instanceof RewriterDataType) || cur.isLiteral())
				return;

			if (namedVariables.put(cur.getId(), cur) != null)
				throw new IllegalArgumentException("Duplicate variable name: " + cur.toParsableString(RuleContext.currentContext));
		});

		Map<RewriterStatement, RewriterStatement> assoc = new DualHashBidiMap<>();

		canonicalForm.forEachPostOrder((cur, parent, pIdx) -> {
			if (!(cur instanceof RewriterDataType) || cur.isLiteral())
				return;

			RewriterStatement ref = namedVariables.get(cur.getId());

			if (ref == null)
				throw new IllegalArgumentException("Unknown variable reference name '" + cur.getId() + "' in: " + cur.toParsableString(RuleContext.currentContext));

			if (reversed)
				assoc.put(cur, ref);
			else
				assoc.put(ref, cur);
		});

		namedVariables.values().forEach(ref -> {
			if (reversed) {
				if (!assoc.containsValue(ref))
					ref.rename("?");
			} else {
				if (!assoc.containsKey(ref))
					ref.rename("?");
			}
		});

		// TODO: If there are some dead references, replace it with an any.<TYPE>() function
		// TODO: Or: just replace var id with '?' to signalize that there is something weird happening
		//if (namedVariables.size() != assoc.size())
		//	throw new IllegalArgumentException("Some variables are not referenced!");

		return assoc;
	}
}
