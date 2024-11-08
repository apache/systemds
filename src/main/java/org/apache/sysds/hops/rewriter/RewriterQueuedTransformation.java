package org.apache.sysds.hops.rewriter;

import org.jetbrains.annotations.NotNull;

public class RewriterQueuedTransformation {
	final RewriterInstruction root;
	final RewriterRuleSet.ApplicableRule rule;

	public RewriterQueuedTransformation(RewriterInstruction root, RewriterRuleSet.ApplicableRule rule) {
		this.root = root;
		this.rule = rule;
	}

	/*@Override
	public int compareTo(@NotNull RewriterQueuedTransformation o) {
		return Long.compare(root.getCost(), o.root.getCost());
	}*/
}
