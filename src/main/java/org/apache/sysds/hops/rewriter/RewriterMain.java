package org.apache.sysds.hops.rewriter;

import org.junit.Rule;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.stream.Collectors;

public class RewriterMain {

	private static RewriterRuleSet ruleSet;
	private static RewriterRule distrib;
	private static RewriterRule commutMul;

	static {
		RewriterRule ruleAddCommut = new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.setUnidirectional(true)
				.withInstruction("+")
					.addOp("a")
						.ofType("float")
					.addOp("b")
						.ofType("float")
					.as("a+b")
					.asRootInstruction()
				.toInstruction("+")
					.addExistingOp("b")
					.addExistingOp("a")
					.as("b+a")
					.asRootInstruction()
				.build();
		RewriterRule ruleAddAssoc = new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.setUnidirectional(false)
				.withInstruction("+")
					.addOp("a")
						.ofType("float")
					.addOp("b")
						.ofType("float")
					.as("a+b")
				.withInstruction("+")
					.addExistingOp("a+b")
					.addOp("c")
						.ofType("float")
					.asRootInstruction()
				.toInstruction("+")
					.addExistingOp("b")
					.addExistingOp("c")
					.as("b+c")
				.toInstruction("+")
					.addExistingOp("a")
					.addExistingOp("b+c")
					.asRootInstruction()
				.build();
		RewriterRule ruleMulCommut = new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.setUnidirectional(true)
				.withInstruction("*")
					.addOp("a")
						.ofType("float")
					.addOp("b")
						.ofType("float")
					.as("a*b")
					.asRootInstruction()
				.toInstruction("*")
					.addExistingOp("b")
					.addExistingOp("a")
					.as("b*a")
					.asRootInstruction()
				.build();
		RewriterRule ruleMulAssoc = new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.setUnidirectional(false)
				.withInstruction("*")
					.addOp("a")
						.ofType("float")
					.addOp("b")
						.ofType("float")
					.as("a*b")
				.withInstruction("*")
					.addExistingOp("a*b")
					.addOp("c")
						.ofType("float")
					.asRootInstruction()
				.toInstruction("*")
					.addExistingOp("b")
					.addExistingOp("c")
					.as("b*c")
				.toInstruction("*")
					.addExistingOp("a")
					.addExistingOp("b*c")
					.asRootInstruction()
				.build();
		RewriterRule ruleDistrib = new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.setUnidirectional(false)
				.withInstruction("*")
					.addOp("a")
						.ofType("float")
					.addOp("c")
						.ofType("float")
					.as("a*c")
				.withInstruction("*")
					.addOp("b")
						.ofType("float")
					.addExistingOp("c")
					.as("b*c")
				.withInstruction("+")
					.addExistingOp("a*c")
					.addExistingOp("b*c")
					.asRootInstruction()
				.toInstruction("+")
					.addExistingOp("a")
					.addExistingOp("b")
					.as("a+b")
				.toInstruction("*")
					.addExistingOp("a+b")
					.addExistingOp("c")
					.asRootInstruction()
				.build();

		RewriterRule ruleOneElement = new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.setUnidirectional(false)
					.withDataType("a", "float")
				.toInstruction("*")
					.addOp("1")
						.ofType("float")
						.asLiteral(1.0f)
					.addExistingOp("a")
					.as("1*a")
					.asRootInstruction()
				.build();


		distrib = ruleDistrib;
		commutMul = ruleMulCommut;

		ArrayList<RewriterRule> rules = new ArrayList<>();
		rules.add(ruleAddCommut);
		rules.add(ruleAddAssoc);
		rules.add(ruleMulCommut);
		rules.add(ruleMulAssoc);
		rules.add(ruleDistrib);
		//rules.add(ruleOneElement);

		ruleSet = new RewriterRuleSet(RuleContext.floatArithmetic, rules);
	}

	/*public static void main(String[] args) {
		System.out.println(ruleSet);

		RewriterInstruction instr = (RewriterInstruction)new RewriterRuleBuilder(RuleContext.floatArithmetic)
				.asDAGBuilder()
				.withInstruction("*")
					.addOp("c")
						.ofType("float")
					.addOp("a")
						.ofType("float")
					.as("c*a")
				.withInstruction("*")
					.addOp("b")
						.ofType("float")
					.addExistingOp("c")
					.as("b*c")
				.withInstruction("*")
					.addExistingOp("a")
					.addExistingOp("c")
					.as("a*c")
				.withInstruction("*")
					.addOp("d")
						.ofType("float")
					.addExistingOp("a")
					.as("d*a")
				.withInstruction("+")
					.addExistingOp("c*a")
					.addExistingOp("b*c")
					.as("par1")
				.withInstruction("+")
					.addExistingOp("par1")
					.addExistingOp("a*c")
					.as("par2")
				.withInstruction("+")
					.addExistingOp("par1")
					.addExistingOp("par2")
					.as("par3")
				.withInstruction("+")
					.addExistingOp("par3")
					.addExistingOp("d*a")
					.asRootInstruction()
				.buildDAG();

		RewriterStatement optimum = instr;
		long optimalCost = instr.getCost();

		RewriterDatabase db = new RewriterDatabase();
		db.insertEntry(RuleContext.floatArithmetic, instr);

		long time = System.currentTimeMillis();

		ArrayList<RewriterRuleSet.ApplicableRule> applicableRules = ruleSet.findApplicableRules(instr);
		PriorityQueue<RewriterInstruction> queue = new PriorityQueue<>();//applicableRules.stream().map(r -> new RewriterQueuedTransformation(instr, r)).sorted().collect(Collectors.toCollection(PriorityQueue::new));
		queue.add((RewriterInstruction) optimum);

		RewriterInstruction current = queue.poll();
		long insertTime = 0;
		long findApplicableRulesTime = 0;
		HashSet<Integer> hashes = new HashSet<>();

		for (int i = 0; i < 1000000 && current != null && queue.size() < 1500000; i++) {
			ArrayList<RewriterRuleSet.ApplicableRule> rules = ruleSet.findApplicableRules(current);
			insertTime = 0;
			findApplicableRulesTime = 0;
			long total = System.nanoTime();
			long trans = 0;
			//System.out.println("Match size: " + current.rule.matches.size());
			for (RewriterRuleSet.ApplicableRule rule : rules) {
				for (RewriterStatement.MatchingSubexpression match : rule.matches) {
					long delta = System.nanoTime();
					RewriterStatement transformed = rule.forward ? rule.rule.applyForward(match, current, false) : rule.rule.applyBackward(match, current, false);
					hashes.add(transformed.hashCode());
					trans += System.nanoTime() - delta;

					delta = System.nanoTime();
					if (!db.insertEntry(RuleContext.floatArithmetic, transformed)) {
						//System.out.println("Skip: " + transformed);
						//System.out.println("======");
						insertTime += System.nanoTime() - delta;
						break; // Then this DAG has already been visited
					}
					insertTime += System.nanoTime() - delta;

					long newCost = transformed.getCost();
					if (newCost < optimalCost) {
						System.out.println("\rFound reduction: " + current + " => " + transformed);
						System.out.println("\tusing " + (rule.forward ? rule.rule.getStmt1() + " => " + rule.rule.getStmt2() : rule.rule.getStmt2() + " => " + rule.rule.getStmt1()));
						optimalCost = newCost;
						optimum = transformed;
					}

					delta = System.nanoTime();
					if (transformed instanceof RewriterInstruction) {
						queue.add((RewriterInstruction)transformed);
					}
				}

				total = System.nanoTime() - total;

				if (i % 100 == 0)
					System.out.print("\r" + db.size() + " unique graphs (Opt: " + optimum + ", Cost: " + optimalCost + ", queueSize: " + queue.size() + ")");
			}

			current = queue.poll();
		}

		System.out.println();
		System.out.println("All possible transformations found in " + (System.currentTimeMillis() - time) + "ms");
		System.out.println("Original graph: " + instr);
		System.out.println("Original cost: " + instr.getCost());
		System.out.println("Optimum: " + optimum);
		System.out.println("Cost: " + optimalCost);
		System.out.println("Unique hashes: " + hashes.size());
	}*/
}
