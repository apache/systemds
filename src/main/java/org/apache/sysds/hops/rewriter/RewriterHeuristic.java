package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableBoolean;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

public class RewriterHeuristic implements RewriterHeuristicTransformation {
	private final RewriterRuleSet ruleSet;
	private final Function<RewriterStatement, RewriterStatement> f;
	private final boolean accelerated;

	public RewriterHeuristic(RewriterRuleSet ruleSet) {
		this(ruleSet, true);
	}

	public RewriterHeuristic(RewriterRuleSet ruleSet, boolean accelerated) {
		this.ruleSet = ruleSet;
		this.accelerated = accelerated;
		this.f = null;
	}

	public RewriterHeuristic(Function<RewriterStatement, RewriterStatement> f) {
		this.ruleSet = null;
		this.accelerated = false;
		this.f = f;
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
		if (f != null)
			return f.apply(currentStmt);

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

		for (int i = 0; i < 500 && rule != null; i++) {
			//System.out.println("Pre-apply: " + rule.rule.getName());
			/*if (currentStmt.toParsableString(ruleSet.getContext()).equals("%*%(X,[](B,1,ncol(X),1,ncol(B)))"))
				System.out.println("test");*/
			/*System.out.println("Expr: " + rule.matches.get(0).getExpressionRoot().toParsableString(ruleSet.getContext()));
			System.out.println("At: " + rule.matches.get(0).getMatchRoot().toParsableString(ruleSet.getContext()));*/
			currentStmt = rule.rule.apply(rule.matches.get(0), currentStmt, rule.forward, false);
			//System.out.println("Now: " + currentStmt.toParsableString(ruleSet.getContext()));

			//transforms.add(currentStmt.toParsableString(ruleSet.getContext()));

			if (handler != null && !handler.apply(currentStmt, rule.rule)) {
				rule = null;
				break;
			}

			if (!(currentStmt instanceof RewriterInstruction)) {
				rule = null;
				break;
			}

			if (accelerated)
				rule = ruleSet.acceleratedFindFirst(currentStmt);
			else
				throw new IllegalArgumentException("Must use accelerated mode!");//rule = ruleSet.findFirstApplicableRule(current);
		}

		if (rule != null)
			throw new IllegalArgumentException("Expression did not converge:\n" + currentStmt.toParsableString(ruleSet.getContext(), true) + "\nRule: " + rule);

		return currentStmt;
	}

	@Override
	public String toString() {
		return ruleSet.toString();
	}
}
