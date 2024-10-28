package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableBoolean;

import javax.annotation.Nullable;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public class RewriterHeuristic implements RewriterHeuristicTransformation {
	private final RewriterRuleSet ruleSet;
	private final boolean accelerated;
	//private final List<String> desiredProperties;

	public RewriterHeuristic(RewriterRuleSet ruleSet) {
		this(ruleSet, true);
	}

	public RewriterHeuristic(RewriterRuleSet ruleSet, boolean accelerated/*, List<String> desiredProperties*/) {
		this.ruleSet = ruleSet;
		this.accelerated = accelerated;
		//this.desiredProperties = desiredProperties;
	}

	public void forEachRuleSet(Consumer<RewriterRuleSet> consumer, boolean printNames) {
		consumer.accept(ruleSet);
	}

	public RewriterStatement apply(RewriterStatement current) {
		return apply(current, null);
	}

	public RewriterStatement apply(RewriterStatement current, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> handler) {
		return apply(current, handler, new MutableBoolean(false), true);
	}

	public RewriterStatement apply(RewriterStatement currentStmt, @Nullable BiFunction<RewriterStatement, RewriterRule, Boolean> handler, MutableBoolean foundRewrite, boolean print) {
		RuleContext.currentContext = ruleSet.getContext();

		//current.forEachPostOrderWithDuplicates(RewriterUtils.propertyExtractor(desiredProperties, ruleSet.getContext()));

		if (handler != null && !handler.apply(currentStmt, null))
			return currentStmt;

		//if (!(currentStmt instanceof RewriterInstruction))
			//return currentStmt;

		//RewriterInstruction current = (RewriterInstruction) currentStmt;

		RewriterRuleSet.ApplicableRule rule;
		if (accelerated)
			rule = ruleSet.acceleratedFindFirst(currentStmt);
		else
			throw new NotImplementedException("Must use accelerated mode");//rule = ruleSet.findFirstApplicableRule(current);

		if (rule != null)
			foundRewrite.setValue(true);

		while (rule != null) {
			/*System.out.println("Pre-apply: " + rule.rule.getName());
			System.out.println("Expr: " + rule.matches.get(0).getExpressionRoot().toParsableString(ruleSet.getContext()));
			System.out.println("At: " + rule.matches.get(0).getMatchRoot().toParsableString(ruleSet.getContext()));*/
			currentStmt = rule.rule.apply(rule.matches.get(0), currentStmt, rule.forward, false);

			if (handler != null && !handler.apply(currentStmt, rule.rule))
				break;

			if (!(currentStmt instanceof RewriterInstruction))
				break;

			if (accelerated)
				rule = ruleSet.acceleratedFindFirst(currentStmt);
			else
				throw new IllegalArgumentException("Must use accelerated mode!");//rule = ruleSet.findFirstApplicableRule(current);
		}

		return currentStmt;
	}

	@Override
	public String toString() {
		return ruleSet.toString();
	}
}
