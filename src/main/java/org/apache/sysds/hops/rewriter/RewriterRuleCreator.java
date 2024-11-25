package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.mutable.MutableBoolean;
import org.apache.sysds.hops.Hop;
import scala.App;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

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

	public synchronized boolean registerRule(RewriterRule rule, long preCost, long postCost, boolean validateCorrectness, Function<RewriterStatement, RewriterStatement> canonicalFormCreator) {
		// First, we check if an existing rule already applies an equivalent rewrite (cost wise)
		RewriterStatement toTest = rule.getStmt1().nestedCopy(false);

		RewriterStatement newStmt = rule.getStmt2().nestedCopy(false);

		boolean converged = false;
		boolean changed = false;

		List<RewriterRule> appliedRules = new ArrayList<>();

		for (int i = 0; i < 500; i++) {
			RewriterRuleSet.ApplicableRule applicableRule = ruleSet.acceleratedFindFirst(newStmt);

			if (applicableRule == null) {
				converged = true;
				break; // Then we converged
			}

			newStmt = applicableRule.rule.apply(applicableRule.matches.get(0), newStmt, applicableRule.forward, false);
			RewriterUtils.mergeArgLists(newStmt, ctx);
			newStmt = RewriterUtils.foldConstants(newStmt, ctx);
			appliedRules.add(applicableRule.rule);
			changed = true;
		}

		if (!converged)
			throw new IllegalArgumentException("The existing rule-set did not seem to converge for the example: \n" + toTest.toParsableString(ctx, true) + "\n" + String.join("\n", appliedRules.stream().map(rl -> rl.toParsableString(ctx)).collect(Collectors.toList())));

		appliedRules.clear();

		for (int i = 0; i < 500; i++) {
			RewriterRuleSet.ApplicableRule applicableRule = ruleSet.acceleratedFindFirst(toTest);

			if (applicableRule == null) {
				converged = true;
				break; // Then we converged
			}

			toTest = applicableRule.rule.apply(applicableRule.matches.get(0), toTest, applicableRule.forward, false);
			RewriterUtils.mergeArgLists(toTest, ctx);
			toTest = RewriterUtils.foldConstants(toTest, ctx);
			appliedRules.add(applicableRule.rule);
			changed = true;
		}

		if (!converged)
			throw new IllegalArgumentException("The existing rule-set did not seem to converge for the example: \n" + toTest.toParsableString(ctx, true) + "\n" + String.join("\n", appliedRules.stream().map(rl -> rl.toParsableString(ctx)).collect(Collectors.toList())));

		if (newStmt != rule.getStmt2()) {
			// Then the mapping has changed, and we need to
			try {
				postCost = RewriterCostEstimator.estimateCost(newStmt, el -> 2000L, ctx);
			} catch (Exception e) {
				System.err.println("Err in cost from orig: " + rule.getStmt2().toParsableString(ctx));
				System.err.println("NewStmt: " + newStmt.toParsableString(ctx));
				e.printStackTrace();
				return false;
			}
		}

		if (changed) {
			long existingPostCost;

			try {
				existingPostCost = RewriterCostEstimator.estimateCost(toTest, el -> 2000L, ctx);
			} catch (Exception e) {
				System.err.println("Err in cost from orig: " + rule.getStmt1().toParsableString(ctx));
				System.err.println("ToTest: " + toTest.toParsableString(ctx));
				System.err.println("AppliedRules: " + appliedRules);
				e.printStackTrace();
				return false;
			}

			if (existingPostCost <= postCost || preCost >= postCost)
				return false; // Then this rule is not beneficial
		}

		// We might have to rebuild the rule
		if (changed || newStmt != rule.getStmt2()) {
			try {
				rule = createRule(toTest, newStmt, canonicalFormCreator.apply(toTest), canonicalFormCreator.apply(newStmt), ctx);
			} catch (Exception e) {
				System.err.println("Failed to create: " + toTest.toParsableString(ctx) + " => " + newStmt.toParsableString(ctx));
			}
		}

		if (validateCorrectness) {
			// Now, we validate the rule by executing it in the system
			if (!validateRuleCorrectnessAndGains(rule, ctx))
				return false; // Then, either the rule is incorrect or is already implemented
		}

		//System.out.println("Rule is correct!");

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
						throw new IllegalArgumentException("The following rule created a conflict with another rule:\nNew one:\n" + rule + "\t[Cost: " + preCost + " => " + postCost + "]\nExisting:\n" + existingRule + "\t[Cost: " + existingRule.getStmt1().getCost(ctx) + " => " + existingRule.getStmt2().getCost(ctx) + "]");
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
		// TODO: Further checks are needed, especially if the new heuristic converges in all cases
		activeRules.add(rule);

		ruleSet.accelerate();
		return true;
	}

	public RewriterRuleSet getRuleSet() {
		return ruleSet;
	}

	public void throwOutInvalidRules(boolean correctness, boolean relevance) {
		if (!correctness && !relevance)
			return;

		activeRules.removeIf(rule -> (correctness && !validateRuleCorrectness(rule, ctx)) || (relevance && !validateRuleApplicability(rule, ctx)));
		ruleSet.accelerate();
	}






	///// STATIC METHODS /////

	// This runs the rule from expressions
	public static boolean validateRuleCorrectnessAndGains(RewriterRule rule, final RuleContext ctx) {
		return validateRuleCorrectness(rule, ctx) && validateRuleApplicability(rule, ctx);
	}

	public static boolean validateRuleCorrectness(RewriterRule rule, final RuleContext ctx) {
		RewriterUtils.renameIllegalVarnames(ctx, rule.getStmt1(), rule.getStmt2());
		String sessionId = UUID.randomUUID().toString();
		String code = DMLCodeGenerator.generateRuleValidationDML(rule, sessionId);

		MutableBoolean isValid = new MutableBoolean(false);
		DMLExecutor.executeCode(code, DMLCodeGenerator.ruleValidationScript(rule.toParsableString(ctx), sessionId, isValid::setValue));

		return isValid.booleanValue();
	}

	public static boolean validateRuleApplicability(RewriterRule rule, final RuleContext ctx) {
		if (ctx.metaPropagator != null)
			ctx.metaPropagator.apply(rule.getStmt1());

		Set<RewriterStatement> vars = DMLCodeGenerator.getVariables(rule.getStmt1());
		Set<String> varNames = vars.stream().map(RewriterStatement::getId).collect(Collectors.toSet());
		String code2Header = DMLCodeGenerator.generateDMLVariables(vars);
		String code2 = code2Header + "\nresult = " + DMLCodeGenerator.generateDML(rule.getStmt1());

		boolean isMatrix = rule.getStmt1().getResultingDataType(ctx).equals("MATRIX");

		if (isMatrix)
			code2 += "\nprint(lineage(result))";
		else
			code2 += "\nprint(lineage(as.matrix(result)))";

		MutableBoolean isRelevant = new MutableBoolean(false);

		RewriterRuntimeUtils.attachHopInterceptor(prog -> {
			Hop hop;

			if (isMatrix)
				hop = prog.getStatementBlocks().get(0).getHops().get(0).getInput(0).getInput(0);
			else
				hop =  prog.getStatementBlocks().get(0).getHops().get(0).getInput(0).getInput(0).getInput(0);

			RewriterStatement stmt = RewriterRuntimeUtils.buildDAGFromHop(hop, 1000, true, ctx);

			if (stmt == null)
				return false;

			Map<String, RewriterStatement> nameAssocs = new HashMap<>();
			// Find the variables that are actually leafs in the original rule
			stmt.forEachPreOrder(cur -> {
				for (int i = 0; i < cur.getOperands().size(); i++) {
					RewriterStatement child = cur.getChild(i);

					if (varNames.contains(child.getId())) {
						RewriterStatement assoc = nameAssocs.get(child.getId());

						if (assoc == null) {
							assoc = new RewriterDataType().as(child.getId()).ofType(child.getResultingDataType(ctx)).consolidate(ctx);

							Long ncol = (Long) child.getMeta("_actualNCol");
							Long nrow = (Long) child.getMeta("_actualNRow");

							if (ncol != null)
								assoc.unsafePutMeta("_actualNCol", ncol);

							if (nrow != null)
								assoc.unsafePutMeta("_actualNRow", nrow);

							nameAssocs.put(child.getId(), assoc);
						}

						cur.getOperands().set(i, assoc);
					}
				}

				return true;
			}, false);

			stmt = RewriterRuntimeUtils.populateDataCharacteristics(stmt, ctx);
			stmt = ctx.metaPropagator.apply(stmt);

			stmt = stmt.nestedCopyOrInject(new HashMap<>(), mstmt -> {
				if (mstmt.isInstruction() && (mstmt.trueInstruction().equals("ncol") || mstmt.trueInstruction().equals("nrow")))
					return RewriterStatement.literal(ctx, DMLCodeGenerator.MATRIX_DIMS);
				return null;
			});

			stmt.prepareForHashing();
			stmt.recomputeHashCodes(ctx);

			Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();

			RewriterStatement stmt1ReplaceNCols = rule.getStmt1().nestedCopyOrInject(createdObjects, mstmt -> {
				if (mstmt.isInstruction() && (mstmt.trueInstruction().equals("ncol") || mstmt.trueInstruction().equals("nrow")))
					return RewriterStatement.literal(ctx, DMLCodeGenerator.MATRIX_DIMS);
				return null;
			});

			stmt1ReplaceNCols.prepareForHashing();
			stmt1ReplaceNCols.recomputeHashCodes(ctx);

			Set<RewriterStatement> mVars = vars.stream().map(createdObjects::get).collect(Collectors.toSet());

			RewriterStatement.MatcherContext mCtx  = RewriterStatement.MatcherContext.exactMatch(ctx, stmt, stmt1ReplaceNCols);
			if (stmt1ReplaceNCols.match(mCtx)) {
				// Check if also the right variables are associated
				boolean assocsMatching = true;
				//DMLExecutor.println(mCtx.getDependencyMap());
				for (RewriterStatement var : mVars) {
					RewriterStatement assoc = mCtx.getDependencyMap().get(var.isInstruction() ? var.getChild(0) : var);

					if (!assoc.getId().equals(var.getId())) {
						assocsMatching = false;
						break;
					}
				}

				if (assocsMatching) {
					// Then the rule matches, meaning that the statement is not rewritten by SystemDS
					isRelevant.setValue(true);
				}
			}

			// TODO: Maybe we can still rewrite the new graph if it still has less cost

			// TODO: Evaluate cost and if our rule can still be applied
			return false; // The program should not be executed as we just want to extract any rewrites that are applied to the current statement
		});

		DMLExecutor.executeCode(code2, true);
		RewriterRuntimeUtils.detachHopInterceptor();

		return isRelevant.booleanValue();
	}

	public static RewriterRule createRule(RewriterStatement from, RewriterStatement to, RewriterStatement canonicalForm1, RewriterStatement canonicalForm2, final RuleContext ctx) {
		from = from.nestedCopy(true);
		to = to.nestedCopy(true);
		Map<RewriterStatement, RewriterStatement> assocs = getAssociations(from, to, canonicalForm1, canonicalForm2, ctx);

		// Now, we replace all variables with a common element
		from.forEachPreOrder((cur, pred) -> {
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
		}, false);

		RewriterRule rule = new RewriterRuleBuilder(ctx, "Autogenerated rule").setUnidirectional(true).completeRule(from, to).build();
		return rule;
	}

	private static Map<RewriterStatement, RewriterStatement> getAssociations(RewriterStatement from, RewriterStatement to, RewriterStatement canonicalFormFrom, RewriterStatement canonicalFormTo, final RuleContext ctx) {
		Map<RewriterStatement, RewriterStatement> fromCanonicalLink = getAssociationToCanonicalForm(from, canonicalFormFrom, true, ctx);
		Map<RewriterStatement, RewriterStatement> toCanonicalLink = getAssociationToCanonicalForm(to, canonicalFormTo, true, ctx);

		RewriterStatement.MatcherContext matcher = RewriterStatement.MatcherContext.exactMatch(ctx, canonicalFormTo, canonicalFormFrom);
		canonicalFormFrom.match(matcher);

		Map<RewriterStatement, RewriterStatement> assocs = new HashMap<>();
		matcher.getDependencyMap().forEach((k, v) -> {
			if (k.isLiteral())
				return;

			RewriterStatement newKey = fromCanonicalLink.get(k);
			RewriterStatement newValue = toCanonicalLink.get(v);

			if (newKey == null || newValue == null)
				return;//throw new IllegalArgumentException("Null reference detected: " + k + ", " + v + "\nFromCanonicalLink: " + fromCanonicalLink + "\nToCanonicalLink: " + toCanonicalLink);

			assocs.put(newKey, newValue);
		});

		return assocs;
	}

	private static Map<RewriterStatement, RewriterStatement> getAssociationToCanonicalForm(RewriterStatement stmt, RewriterStatement canonicalForm, boolean reversed, final RuleContext ctx) {
		// We identify all associations by their names
		// If there are name collisions, this does not work
		Map<String, RewriterStatement> namedVariables = new HashMap<>();
		stmt.forEachPostOrder((cur, pred) -> {
			if (!(cur instanceof RewriterDataType) || cur.isLiteral())
				return;

			if (namedVariables.put(cur.getId(), cur) != null)
				throw new IllegalArgumentException("Duplicate variable name: " + cur.toParsableString(RuleContext.currentContext) + "\nEntire statement:\n" + stmt.toParsableString(ctx) + "\nRaw: " + stmt);
		}, false);

		Map<RewriterStatement, RewriterStatement> assoc = new DualHashBidiMap<>();

		canonicalForm.forEachPostOrder((cur, pred) -> {
			if (!(cur instanceof RewriterDataType) || cur.isLiteral())
				return;

			RewriterStatement ref = namedVariables.get(cur.getId());

			if (ref == null)
				throw new IllegalArgumentException("Unknown variable reference name '" + cur.getId() + "' in: " + cur.toParsableString(RuleContext.currentContext));

			if (reversed)
				assoc.put(cur, ref);
			else
				assoc.put(ref, cur);
		}, false);

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
