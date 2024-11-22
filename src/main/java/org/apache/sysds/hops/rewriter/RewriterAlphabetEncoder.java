package org.apache.sysds.hops.rewriter;

import com.google.protobuf.Internal;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableInt;
import org.apache.spark.internal.config.R;
import org.apache.sysds.runtime.compress.workload.Op;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class RewriterAlphabetEncoder {
	private static final List<String> ALL_TYPES = List.of("MATRIX", "FLOAT");
	private static final List<String> MATRIX = List.of("MATRIX");

	private static Operand[] instructionAlphabet = new Operand[] {
			null,
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
			new Operand("log", 1, MATRIX),

			// Fused operators
			new Operand("1-*", 2, MATRIX), 			// TODO: We have to include literals in the search
			new Operand("log_nz", 1, MATRIX)			// TODO: We have to include literals in the search
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

	// To include structures like row/column vectors etc.
	public static List<RewriterStatement> buildAssertionVariations(RewriterStatement root, final RuleContext ctx, boolean increasedVariance) {
		List<RewriterStatement> interestingLeaves = new ArrayList<>();
		root.forEachPreOrder(cur -> {
			if (!cur.isInstruction() && !cur.isLiteral() && cur.getResultingDataType(ctx).equals("MATRIX"))
				interestingLeaves.add(cur);
			return true;
		}, true);

		if (interestingLeaves.isEmpty())
			return List.of(root);

		List<RewriterStatement> out = new ArrayList<>();
		out.add(root);

		for (int i = 0; i < interestingLeaves.size(); i++) {
			RewriterStatement from = interestingLeaves.get(i);
			RewriterStatement rv = createVectorizedStatement(root, from, true);
			if (ctx.metaPropagator != null)
				rv = ctx.metaPropagator.apply(rv);
			out.add(rv);
			RewriterStatement cv = createVectorizedStatement(root, from, false);
			if (ctx.metaPropagator != null)
				cv = ctx.metaPropagator.apply(cv);
			out.add(cv);

			for (int j = i + 1; j < interestingLeaves.size(); j++) {
				RewriterStatement to = interestingLeaves.get(i);
				Map<RewriterStatement, Boolean> map = new HashMap<>();
				map.put(from, false);
				map.put(to, false);
				out.add(createVectorizedStatements(root, map));
				map.put(from, true);
				out.add(createVectorizedStatements(root, map));
				map.put(to, true);
				out.add(createVectorizedStatements(root, map));
				map.put(from, false);
				out.add(createVectorizedStatements(root, map));
			}
		}

		if (ctx.metaPropagator != null)
			return out.stream().map(stmt -> ctx.metaPropagator.apply(stmt)).collect(Collectors.toList());

		return out;
	}

	private static RewriterStatement createVector(RewriterStatement of, boolean rowVector, Map<RewriterStatement, RewriterStatement> createdObjects) {
		// TODO: Why is it necessary to discard the old DataType?
		RewriterStatement mCpy = new RewriterDataType().as(of.getId()).ofType(of.getResultingDataType(ctx)).consolidate(ctx);
		RewriterStatement nRowCol = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction(rowVector ? "nrow" : "ncol").withOps(mCpy).consolidate(ctx);
		createdObjects.put(of, mCpy);
		return new RewriterInstruction()
				.as(UUID.randomUUID().toString())
				.withInstruction("[]")
				.withOps(
						mCpy,
						RewriterStatement.literal(ctx, 1L),
						rowVector ? nRowCol : RewriterStatement.literal(ctx, 1L),
						RewriterStatement.literal(ctx, 1L),
						rowVector ? RewriterStatement.literal(ctx, 1L) : nRowCol)
				.consolidate(ctx);
	}

	private static RewriterStatement createVectorizedStatement(RewriterStatement root, RewriterStatement of, boolean rowVector) {
		HashMap<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
		RewriterStatement out = root.nestedCopyOrInject(createdObjects, stmt -> {
			if (stmt.equals(of))
				return createVector(of, rowVector, createdObjects);

			return null;
		});

		return out;
	}

	private static RewriterStatement createVectorizedStatements(RewriterStatement root, Map<RewriterStatement, Boolean> of) {
		HashMap<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();

		RewriterStatement out = root.nestedCopyOrInject(createdObjects, stmt -> {
			if (!stmt.isInstruction() && !stmt.isLiteral() && stmt.getResultingDataType(ctx).equals("MATRIX")) {
				Boolean rowVector = of.get(stmt);

				if (rowVector != null)
					return createVector(stmt, rowVector, createdObjects);
			}

			return null;
		});

		return out;
	}

	// Builds variations of the same graph (e.g. +(A,B) -> +(A,A))
	public static List<RewriterStatement> buildVariations(RewriterStatement root, final RuleContext ctx) {
		List<RewriterStatement> interestingLeaves = new ArrayList<>();
		root.forEachPreOrder(cur -> {
			if (!cur.isInstruction() && !cur.isLiteral() && cur.getResultingDataType(ctx).equals("MATRIX"))
				interestingLeaves.add(cur);
			return true;
		}, true);

		if (interestingLeaves.size() < 2)
			return List.of(root);

		List<RewriterStatement> out = new ArrayList<>();
		out.add(root);

		for (int i = 0; i < interestingLeaves.size(); i++) {
			RewriterStatement to = interestingLeaves.get(i);
			for (int j = i + 1; j < interestingLeaves.size(); j++) {
				RewriterStatement from = interestingLeaves.get(j);
				HashMap<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();
				RewriterStatement toCpy = new RewriterDataType().as(to.getId()).ofType(to.getResultingDataType(ctx)).consolidate(ctx);
				createdObjects.put(from, toCpy);
				createdObjects.put(to, toCpy);
				RewriterStatement cpy = root.nestedCopyOrInject(createdObjects, stmt -> null);
				if (ctx.metaPropagator != null)
					cpy = ctx.metaPropagator.apply(cpy);
				out.add(cpy);
				//System.out.println("HERE:" + out.get(out.size()-1));
			}
		}

		return out;
	}

	public static List<RewriterStatement> buildAllPossibleDAGs(List<Operand> operands, final RuleContext ctx, boolean rename) {
		if (operands == null)
			return Collections.emptyList();

		RewriterAlphabetEncoder.ctx = ctx;

		List<RewriterStatement> allStmts = recursivelyFindAllCombinations(operands);

		if (rename)
			allStmts.forEach(RewriterAlphabetEncoder::rename);

		if (ctx.metaPropagator != null)
			allStmts = allStmts.stream().map(stmt -> ctx.metaPropagator.apply(stmt)).collect(Collectors.toList());

		return allStmts;
	}

	private static List<RewriterStatement> recursivelyFindAllCombinations(List<Operand> operands) {
		if (operands.isEmpty())
			return Stream.concat(ALL_TYPES.stream().map(t -> new RewriterDataType().as(UUID.randomUUID().toString()).ofType(t).consolidate(ctx)), Stream.of(RewriterStatement.literal(ctx, 1.0D), RewriterStatement.literal(ctx, 0.0D))).collect(Collectors.toList());

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
		//System.out.println("StmtIdx: " + stmt);

		for (int i = 0; i < instructions.length; i++) {
			/*System.out.println("Idx: " + i);
			System.out.println("digits[" + i + "]: " + instructions[i]);
			System.out.println("As op: " + instructionAlphabet[instructions[i]]);*/
			Operand toAdd = instructionAlphabet[instructions[i]];
			if (toAdd == null)
				return null;
			out.add(toAdd);
		}

		return out;
	}

	public static int[] fromBaseNNumber(int l, int n) {
		if (l == 0)
			return new int[0];

		// We put 1 as the last bit to signalize end of sequence
		/*int m = Integer.numberOfTrailingZeros(Integer.highestOneBit(l));
		int maxRepr = 1 << (m - 1);
		l = l ^ (1 << m);

		System.out.println("Bin: " + Integer.toBinaryString(l));
		System.out.println("m: " + m);
		System.out.println("l: " + l);*/

		int numDigits = (int)(Math.log(l) / Math.log(n)) + 1;
		int[] digits = new int[numDigits];

		for (int i = numDigits - 1; i >= 0; i--) {
			//System.out.println(l + " % " + n);
			digits[i] = l % n;
			l = l / n;
		}

		/*System.out.println("numDigits: " + numDigits);
		System.out.println("digits[0]: " + digits[0]);*/

		return digits;
	}

	public static int toBaseNNumber(int[] digits, int n) {
		if (digits.length == 0)
			throw new IllegalArgumentException();

		int multiplicator = 1;
		int out = 0;
		//int maxPossible = 0;

		for (int i = digits.length - 1; i >= 0; i--) {
			out += multiplicator * digits[i];
			//maxPossible += multiplicator * (n - 1);
			multiplicator *= n;
		}

		/*int m = Integer.numberOfTrailingZeros(Integer.highestOneBit(maxPossible));
		out |= (1 << m);*/

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
