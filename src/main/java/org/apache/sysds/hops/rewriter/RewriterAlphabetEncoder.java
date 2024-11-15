package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableInt;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

public class RewriterAlphabetEncoder {
	private static final List<String> ALL_TYPES = List.of("MATRIX", "FLOAT");
	private static final List<String> MATRIX = List.of("MATRIX");

	private static Operand[] instructionAlphabet = new Operand[] {
			new Operand("+", 2, ALL_TYPES),
			new Operand("-", 2, ALL_TYPES),
			new Operand("*", 2, ALL_TYPES),
			new Operand("/", 2, ALL_TYPES),
			new Operand("%*%", 2, ALL_TYPES),

			new Operand("sum", 1, MATRIX),
			new Operand("t", 1, MATRIX),
			new Operand("trace", 1, MATRIX),
			new Operand("rowSums", 1, MATRIX),
			new Operand("colSums", 1, MATRIX),
			new Operand("max", 1, MATRIX),
			new Operand("min", 1, MATRIX),
	};

	private static String[] varNames = new String[] {
		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"
	};

	private static RuleContext ctx;

	/*private static List<String> allPossibleTypes(Operand op, int argNum) {
		if (op == null)
			return List.of("MATRIX", "FLOAT");

		switch (op.op) {
			case "+":
				return List.of("MATRIX", "FLOAT");
			case "-":
				return List.of("MATRIX", "FLOAT");
			case "*":
				return List.of("MATRIX", "FLOAT");
			case "/":
				return List.of("MATRIX", "FLOAT");
		}

		throw new NotImplementedException();
	}*/

	public static void rename(RewriterStatement stmt) {
		Set<RewriterStatement> namedVars = new HashSet<>();

		stmt.forEachPostOrder((cur, pred) -> {
			if (!cur.isInstruction() && !cur.isLiteral()) {
				if (!namedVars.contains(cur)) {
					if (cur.getResultingDataType(ctx).equals("MATRIX"))
						cur.rename(varNames[namedVars.size()]);
					else
						cur.rename(varNames[namedVars.size()].toLowerCase());

					namedVars.add(cur);
				}
			}
		}, false);
	}

	public static List<RewriterStatement> buildAllPossibleDAGs(List<Operand> operands, final RuleContext ctx, boolean rename) {
		RewriterAlphabetEncoder.ctx = ctx;

		List<RewriterStatement> allStmts = recursivelyFindAllCombinations(operands);

		if (rename)
			allStmts.forEach(RewriterAlphabetEncoder::rename);

		if (ctx.metaPropagator != null)
			return allStmts.stream().map(stmt -> ctx.metaPropagator.apply(stmt)).collect(Collectors.toList());
		else
			return allStmts;
	}

	private static List<RewriterStatement> recursivelyFindAllCombinations(List<Operand> operands) {
		if (operands.isEmpty())
			return ALL_TYPES.stream().map(t -> new RewriterDataType().as(UUID.randomUUID().toString()).ofType(t).consolidate(ctx)).collect(Collectors.toList());

		int nOps = operands.get(0).numArgs;
		int[] slices = new int[nOps-1];

		List<RewriterStatement> possibleStmts = new ArrayList<>();

		forEachSlice(1, 0, operands.size()+1, slices, () -> {
			List<List<RewriterStatement>> cartesianBuilder = new ArrayList<>();

			for (int i = 0; i < nOps; i++) {
				int lIdx = i == 0 ? 1 : slices[i-1];
				int uIdx = i == slices.length ? operands.size() : slices[i];

				List<Operand> view;
				if (lIdx == uIdx)
					view = Collections.emptyList();
				else
					view = operands.subList(lIdx, uIdx);

				List<RewriterStatement> combs = recursivelyFindAllCombinations(view);
				if (combs.isEmpty())
					return; // Then no subgraph can be created from that order

				cartesianBuilder.add(combs);
			}

			RewriterStatement[] stack = new RewriterStatement[nOps];
			RewriterUtils.cartesianProduct(cartesianBuilder, stack, mStack -> {
				try {
					possibleStmts.add(new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction(operands.get(0).op).withOps(stack).consolidate(ctx));
				} catch (Exception e) {
					// Might fail, as there could be wrong types
					//e.printStackTrace();
				}
				return true; // Should continue
			});
		});

		return possibleStmts;
	}

	private static void forEachSlice(int startIdx, int pos, int maxIdx, int[] slices, Runnable trigger) {
		if (pos >= slices.length) {
			trigger.run();
			return;
		}

		for (int idx = startIdx; idx < maxIdx; idx++) {
			slices[pos] = idx;

			if (pos != slices.length-1) {
				forEachSlice(idx, pos+1, maxIdx, slices, trigger);
			} else {
				trigger.run();
			}
		}
	}

	public static List<Operand> decodeOrderedStatements(int stmt) {
		int[] instructions = fromBaseNNumber(stmt, instructionAlphabet.length);
		List<Operand> out = new ArrayList<>(instructions.length);

		for (int i = 0; i < instructions.length; i++)
			out.add(instructionAlphabet[instructions[i]]);

		return out;
	}

	public static int[] fromBaseNNumber(int l, int n) {
		if (l == 0)
			return new int[0];

		// We put 1 as the last bit to signalize end of sequence
		int m = Integer.numberOfTrailingZeros(Integer.highestOneBit(l));
		int maxRepr = 1 << (m - 1);
		l = l ^ (1 << m);

		int numDigits = (int)(Math.log(maxRepr) / Math.log(n)) + 1;
		int[] digits = new int[numDigits];

		for (int i = numDigits - 1; i >= 0; i--) {
			digits[i] = l % n;
			l = l / n;
		}

		return digits;
	}

	public static int toBaseNNumber(int[] digits, int n) {
		if (digits.length == 0)
			throw new IllegalArgumentException();

		int multiplicator = 1;
		int out = 0;
		int maxPossible = 0;

		for (int i = digits.length - 1; i >= 0; i--) {
			out += multiplicator * digits[i];
			maxPossible += multiplicator * (n - 1);
			multiplicator *= n;
		}

		int m = Integer.numberOfTrailingZeros(Integer.highestOneBit(maxPossible));
		out |= (1 << m);

		return out;
	}

	public static final class Operand {
		public final String op;
		public final int numArgs;
		public final List<String> supportedTypes;
		public Operand(String op, int numArgs, List<String> supportedTypes) {
			this.op = op;
			this.numArgs = numArgs;
			this.supportedTypes = supportedTypes;
		}

		public String toString() {
			return op;
		}
	}
}
