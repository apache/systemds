package org.apache.sysds.hops.rewriter;

import java.util.HashSet;

public class RewriterExamples {
	public static RewriterInstruction selectionPushdownExample1() {
		return (RewriterInstruction)new RewriterRuleBuilder(RuleContext.selectionPushdownContext)
				.asDAGBuilder()
				.withInstruction("RowSelectPushableBinaryInstruction") // This is more a class of instructions
					.addOp("A")
						.ofType("MATRIX")
					.addOp("B")
						.ofType("MATRIX")
					.as("A + B")
				.withInstruction("rowSelect")
					.addExistingOp("A + B")
					.addOp("i")
						.ofType("INT")
					.addOp("j")
						.ofType("INT")
					.asRootInstruction()
				.buildDAG();
	}

	public static RewriterInstruction selectionPushdownExample2() {
		return (RewriterInstruction)new RewriterRuleBuilder(RuleContext.selectionPushdownContext)
				.asDAGBuilder()
				.withInstruction("RowSelectPushableBinaryInstruction") // This is more a class of instructions
					.instrMeta("trueName", "+")
					.addOp("H")
						.ofType("MATRIX")
					.addOp("K")
						.ofType("MATRIX")
					.as("H + K")
				.withInstruction("rowSelect")
					.addExistingOp("H + K")
					.addOp("n")
						.ofType("INT")
					.addOp("m")
						.ofType("INT")
					.as("rowSelect(H + K, n, m)")
				.withInstruction("rowSelect")
					.addExistingOp("rowSelect(H + K, n, m)")
					.addOp("a")
						.ofType("INT")
					.addOp("b")
						.ofType("INT")
					.asRootInstruction()
				.buildDAG();
	}

	public static RewriterInstruction selectionPushdownExample3() {
		return (RewriterInstruction)new RewriterRuleBuilder(RuleContext.selectionPushdownContext)
				.asDAGBuilder()
				.withInstruction("+") // This is more a class of instructions
				.instrMeta("trueName", "+")
				.addOp("H")
				.ofType("MATRIX")
				.addOp("K")
				.ofType("MATRIX")
				.as("H + K")
				.withInstruction("rowSelect")
				.instrMeta("trueName", "[,?]")
				.addExistingOp("H + K")
				.addOp("n")
				.ofType("INT")
				.addOp("m")
				.ofType("INT")
				.as("rowSelect(H + K, n, m)")
				.withInstruction("rowSelect")
				.addExistingOp("rowSelect(H + K, n, m)")
				.addOp("a")
				.ofType("INT")
				.addOp("b")
				.ofType("INT")
				.asRootInstruction()
				.buildDAG();
	}

	public static RewriterInstruction selectionPushdownExample4(final RuleContext ctx) {
		return (RewriterInstruction)new RewriterRuleBuilder(ctx)
				.asDAGBuilder()
				.withInstruction("+") // This is more a class of instructions
				.addOp("H")
				.ofType("MATRIX")
				.addOp("K")
				.ofType("MATRIX")
				.as("H + K")
				.withInstruction("+")
				.addExistingOp("H + K")
				.addOp("M")
				.ofType("MATRIX")
				.as("H + K + M")
				.withInstruction("rowSelect")
				.addExistingOp("H + K + M")
				.addOp("n")
				.ofType("INT")
				.addOp("m")
				.ofType("INT")
				.as("rowSelect(H + K, n, m)")
				.withInstruction("colSelect")
				.addExistingOp("rowSelect(H + K, n, m)")
				.addOp("k")
				.ofType("INT")
				.addOp("l")
				.ofType("INT")
				.as("ir")
				.withInstruction("rowSelect")
				.addExistingOp("ir")
				.addOp("a")
				.ofType("INT")
				.addOp("b")
				.ofType("INT")
				.as("ir2")
				.withInstruction("index")
				.addExistingOp("ir2")
				.addOp("q").ofType("INT")
				.addOp("r").ofType("INT")
				.addOp("s").ofType("INT")
				.addOp("t").ofType("INT")
				.asRootInstruction()
				.buildDAG();
	}
}
