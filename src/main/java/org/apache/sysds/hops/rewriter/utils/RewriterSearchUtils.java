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

package org.apache.sysds.hops.rewriter.utils;

import org.apache.sysds.hops.rewriter.RewriterDataType;
import org.apache.sysds.hops.rewriter.RewriterInstruction;
import org.apache.sysds.hops.rewriter.RewriterStatement;
import org.apache.sysds.hops.rewriter.RuleContext;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

public class RewriterSearchUtils {
	public static final List<String> ALL_TYPES = List.of("MATRIX", "FLOAT");
	public static final List<String> SCALAR = List.of("FLOAT");
	public static final List<String> MATRIX = List.of("MATRIX");

	public static Operand[] instructionAlphabet = new Operand[] {
			null,
			new Operand("+", 2, ALL_TYPES, ALL_TYPES),
			//new Operand("+", 2, MATRIX, SCALAR),
			//new Operand("+", 2, MATRIX, MATRIX),

			new Operand("-", 2, ALL_TYPES, ALL_TYPES),
			//new Operand("-", 2, MATRIX, SCALAR),
			//new Operand("-", 2, MATRIX, MATRIX),

			new Operand("*", 2, ALL_TYPES, ALL_TYPES),
			//new Operand("*", 2, MATRIX, SCALAR),
			//new Operand("*", 2, MATRIX, MATRIX),

			new Operand("/", 2, ALL_TYPES, ALL_TYPES),
			//new Operand("/", 2, MATRIX, SCALAR),
			//new Operand("/", 2, MATRIX, MATRIX),

			new Operand("%*%", 2, MATRIX, MATRIX),

			new Operand("sum", 1, MATRIX),
			new Operand("*sum", 2, MATRIX, ALL_TYPES), // To have a bigger search space for this instruction combination
			new Operand("t", 1, MATRIX),
			new Operand("rev", 1, MATRIX),
			new Operand("diag", 1, MATRIX),
			new Operand("trace", 1, MATRIX),
			new Operand("rowSums", 1, MATRIX),
			new Operand("colSums", 1, MATRIX),
			new Operand("max", 1, MATRIX),
			new Operand("min", 1, MATRIX),
			new Operand("ncol", 0, true, MATRIX),
			new Operand("nrow", 0, true, MATRIX),
			new Operand("length", 0, true, MATRIX),

			new Operand("!=", 2, ALL_TYPES, ALL_TYPES),
			new Operand("!=0", 1, MATRIX),
			new Operand("0!=", 1, MATRIX),

			new Operand("cast.MATRIX",1, SCALAR),
			new Operand("cast.FLOAT", 1, MATRIX),

			new Operand("1-*", 2, MATRIX, MATRIX),
			new Operand("+*", 3, MATRIX, SCALAR, MATRIX),
			new Operand("-*", 3, MATRIX, SCALAR, MATRIX),
			new Operand("*2", 1, MATRIX),
			new Operand("_nnz", 1, MATRIX),
			new Operand("sumSq", 1, MATRIX),
			new Operand("sq", 1, MATRIX),
			//new Operand("log", 1, MATRIX),

			// constant stuff
			new Operand("c_1+", 1, ALL_TYPES),
			new Operand("c_+1", 1, ALL_TYPES),
			new Operand("c_1-", 1, ALL_TYPES),
			new Operand("c_-1", 1, ALL_TYPES),

			// ncol / nrow / length stuff
			new Operand("c_length*", 1, ALL_TYPES),
			new Operand("c_ncol*", 1, ALL_TYPES),
			new Operand("c_nrow*", 1, ALL_TYPES),

			new Operand("log_nz", 1, MATRIX),

			// Placeholder operators
			new Operand("zero", 0, true),
			new Operand("one", 0, true)
	};

	private static String[] varNames = new String[] {
		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"
	};

	private static RuleContext ctx;

	public static int getMaxSearchNumberForNumOps(int numOps) {
		int out = 1;
		for (int i = 0; i < numOps; i++)
			out *= instructionAlphabet.length;

		return out;
	}

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
	public static List<RewriterStatement> buildAssertionVariations(RewriterStatement root, final RuleContext ctx) {
		List<RewriterStatement> interestingLeaves = new ArrayList<>();
		root.forEachPreOrder(cur -> {
			if (!cur.isInstruction() && !cur.isLiteral() && cur.getResultingDataType(ctx).equals("MATRIX"))
				interestingLeaves.add(cur);
			return true;
		}, true);

		if (interestingLeaves.isEmpty())
			return Collections.emptyList();

		List<RewriterStatement> out = new ArrayList<>();

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

		// Serialize and parse again as there may still be duplicate references
		out = out.stream().map(stmt -> RewriterUtils.parse(stmt.toParsableString(ctx, true), ctx)).collect(Collectors.toList());

		if (ctx.metaPropagator != null)
			return out.stream().map(stmt -> ctx.metaPropagator.apply(stmt)).collect(Collectors.toList());

		return out;
	}

	private static RewriterStatement createVector(RewriterStatement of, boolean rowVector, Map<RewriterStatement, RewriterStatement> createdObjects) {
		// TODO: Why is it necessary to discard the old DataType?
		RewriterStatement mCpy = createdObjects.get(of);

		if (mCpy == null) {
			mCpy = new RewriterDataType().as(of.getId()).ofType(of.getResultingDataType(ctx)).consolidate(ctx);
			createdObjects.put(of, mCpy);
		}
		//RewriterStatement nRowCol = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction(rowVector ? "nrow" : "ncol").withOps(mCpy).consolidate(ctx);
		//createdObjects.put(of, mCpy);
		return new RewriterInstruction()
				.as(of.getId())
				.withInstruction(rowVector ? "rowVec" : "colVec")
				.withOps(mCpy)
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
			return Collections.emptyList();

		List<RewriterStatement> out = new ArrayList<>();

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
			}
		}

		// Serialize and parse again as there may still be duplicate references
		out = out.stream().map(stmt -> RewriterUtils.parse(stmt.toParsableString(ctx, true), ctx)).collect(Collectors.toList());

		return out;
	}

	public static List<RewriterStatement> buildAllPossibleDAGs(List<Operand> operands, final RuleContext ctx, boolean rename) {
		if (operands == null)
			return Collections.emptyList();

		RewriterSearchUtils.ctx = ctx;

		List<RewriterStatement> allStmts = recursivelyFindAllCombinations(operands, null, ALL_TYPES);

		if (rename)
			allStmts.forEach(RewriterSearchUtils::rename);

		if (ctx.metaPropagator != null)
			allStmts = allStmts.stream().map(stmt -> ctx.metaPropagator.apply(stmt)).collect(Collectors.toList());

		// Serialize and parse all statements as there are still duplicate references
		return allStmts.stream().map(stmt -> RewriterUtils.parse(stmt.toParsableString(ctx, true), ctx)).collect(Collectors.toList());
	}

	private static List<RewriterStatement> recursivelyFindAllCombinations(List<Operand> operands, Operand parent, List<String> supportedTypes) {
		if (operands.isEmpty())
			return supportedTypes.stream().map(t -> new RewriterDataType().as(UUID.randomUUID().toString()).ofType(t).consolidate(ctx)).collect(Collectors.toList());

		// Check if op is a placeholder
		Operand op = operands.get(0);

		if (op.isLeaf && operands.size() > 1)
			return Collections.emptyList();

		if (op.op.equals("zero") || op.op.equals("one")) {
			List<RewriterStatement> l = new ArrayList<>(2);
			if (op.op.equals("zero")) {
				if (supportedTypes.contains("FLOAT"))
					l.add(RewriterStatement.literal(ctx, 0.0D));
				if (supportedTypes.contains("MATRIX"))
					l.add(new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("const").withOps(new RewriterDataType().as(UUID.randomUUID().toString()).ofType("MATRIX").consolidate(ctx), RewriterStatement.literal(ctx, 0.0D)).consolidate(ctx));
			} else {
				if (supportedTypes.contains("FLOAT"))
					l.add(RewriterStatement.literal(ctx, 1.0D));

				if (supportedTypes.contains("MATRIX"))
					l.add(new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("const").withOps(new RewriterDataType().as(UUID.randomUUID().toString()).ofType("MATRIX").consolidate(ctx), RewriterStatement.literal(ctx, 1.0D)).consolidate(ctx));
			}

			return l;
		}

		int nOps = operands.get(0).numArgs;

		if (nOps == 0) {
			return List.of(buildStmt(op, null));
		}

		int[] slices = new int[Math.max(nOps-1, 0)];

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

				List<RewriterStatement> combs = recursivelyFindAllCombinations(view, op, op.supportedTypes[i]);

				if (combs.isEmpty())
					return; // Then no subgraph can be created from that order

				cartesianBuilder.add(combs);
			}

			RewriterStatement[] stack = new RewriterStatement[nOps];
			RewriterUtils.cartesianProduct(cartesianBuilder, stack, mStack -> {
				try {
					for (int i = 0; i < stack.length; i++)
						if (!op.supportedTypes[i].contains(stack[i].getResultingDataType(ctx)))
							return true;

					RewriterStatement stmt = buildStmt(operands.get(0), stack);
					possibleStmts.add(stmt);
				} catch (Exception e) {
					// Might fail as there could be wrong types
				}
				return true; // Should continue
			});
		});

		return possibleStmts;
	}

	private static RewriterStatement buildStmt(Operand op, RewriterStatement[] stack) {
		RewriterInstruction stmt = new RewriterInstruction().as(UUID.randomUUID().toString());
		switch (op.op) {
			case "!=0": {
				stmt.withInstruction("!=").addOp(stack[0]).addOp(RewriterStatement.literal(ctx, 0.0D));
				break;
			}
			case "0!=": {
				stmt.withInstruction("!=").addOp(RewriterStatement.literal(ctx, 0.0D)).addOp(stack[0]);
				break;
			}
			case "ncol":
			case "nrow":
			case "length": {
				String actualOp = op.op;
				stmt.withInstruction(actualOp).withOps(new RewriterDataType().as(UUID.randomUUID().toString()).ofType("MATRIX").consolidate(ctx)).consolidate(ctx);
				break;
			}
			case "fncol":
			case "fnrow":
			case "flength": {
				String actualOp = op.op.substring(1);
				stmt.withInstruction(actualOp).withOps(stack).consolidate(ctx);
				stmt = (RewriterInstruction) RewriterStatement.castFloat(ctx, stmt);
				break;
			}
			case "*sum": {
				RewriterStatement old = stmt.withInstruction("sum").withOps(stack[0]).consolidate(ctx);
				stmt = new RewriterInstruction("*", ctx, old, stack[1]);
				break;
			}
			case "c_1+": {
				stmt = new RewriterInstruction("+", ctx, RewriterStatement.literal(ctx, 1.0D), stack[0]);
				break;
			}
			case "c_+1": {
				stmt = new RewriterInstruction("+", ctx, stack[0], RewriterStatement.literal(ctx, 1.0D));
				break;
			}
			case "c_1-": {
				stmt = new RewriterInstruction("-", ctx, RewriterStatement.literal(ctx, 1.0D), stack[0]);
				break;
			}
			case "c_-1": {
				stmt = new RewriterInstruction("-", ctx, stack[0], RewriterStatement.literal(ctx, 1.0D));
				break;
			}
			case "c_length*": {
				stmt = new RewriterInstruction("*", ctx, new RewriterInstruction("length", ctx, new RewriterDataType().as(UUID.randomUUID().toString()).ofType("MATRIX").consolidate(ctx)), stack[0]);
				break;
			}
			case "c_nrow*": {
				stmt = new RewriterInstruction("*", ctx, new RewriterInstruction("nrow", ctx, new RewriterDataType().as(UUID.randomUUID().toString()).ofType("MATRIX").consolidate(ctx)), stack[0]);
				break;
			}
			case "c_col*": {
				stmt = new RewriterInstruction("*", ctx, new RewriterInstruction("ncol", ctx, new RewriterDataType().as(UUID.randomUUID().toString()).ofType("MATRIX").consolidate(ctx)), stack[0]);
				break;
			}
			default: {
				stmt.withInstruction(op.op).withOps(stack);
				break;
			}
		}

		stmt.consolidate(ctx);
		return stmt;
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

		for (int i = 0; i < instructions.length; i++) {
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

		int numDigits = (int)(Math.log(l) / Math.log(n)) + 1;
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

		for (int i = digits.length - 1; i >= 0; i--) {
			out += multiplicator * digits[i];
			multiplicator *= n;
		}

		return out;
	}

	public static List<RewriterStatement> mergeSubtreeCombinations(RewriterStatement stmt, List<Integer> indices, List<List<RewriterStatement>> mList, final RuleContext ctx, int maximumCombinations) {
		if (indices.isEmpty())
			return List.of(stmt);

		List<RewriterStatement> mergedTreeCombinations = new ArrayList<>();
		RewriterUtils.cartesianProduct(mList, new RewriterStatement[mList.size()], stack -> {
			RewriterStatement cpy = stmt.copyNode();
			for (int i = 0; i < stack.length; i++)
				cpy.getOperands().set(indices.get(i), stack[i]);
			cpy.consolidate(ctx);
			cpy.prepareForHashing();
			cpy.recomputeHashCodes(ctx);
			mergedTreeCombinations.add(cpy);
			return mergedTreeCombinations.size() < maximumCombinations;
		});

		return mergedTreeCombinations;
	}

	public static List<RewriterStatement> generateSubtrees(RewriterStatement stmt, final RuleContext ctx, int maximumCombinations) {
		List<RewriterStatement> l = generateSubtrees(stmt, new HashMap<>(), ctx, maximumCombinations);

		if (ctx.metaPropagator != null)
			l.forEach(subtree -> ctx.metaPropagator.apply(subtree));

		return l.stream().map(subtree -> {
			if (ctx.metaPropagator != null)
				subtree = ctx.metaPropagator.apply(subtree);

			subtree.prepareForHashing();
			subtree.recomputeHashCodes(ctx);
			// We return a copy of the tree as there are still duplicate references
			return RewriterUtils.parse(subtree.toParsableString(ctx, true), ctx);
		}).collect(Collectors.toList());
	}

	private static Random rd = new Random();

	private static List<RewriterStatement> generateSubtrees(RewriterStatement stmt, Map<RewriterStatement, List<RewriterStatement>> visited, final RuleContext ctx, int maxCombinations) {
		if (stmt == null)
			return Collections.emptyList();

		RewriterStatement is = stmt;
		List<RewriterStatement> alreadyVisited = visited.get(is);

		if (alreadyVisited != null)
			return alreadyVisited;

		if (stmt.getOperands().size() == 0)
			return List.of(stmt);

		// Scan if operand is not a DataType
		List<Integer> indices = new ArrayList<>();
		for (int i = 0; i < stmt.getOperands().size(); i++) {
			if (stmt.getChild(i).isInstruction() || stmt.getChild(i).isLiteral())
				indices.add(i);
		}

		int n = indices.size();
		int totalSubsets = 1 << n;

		List<RewriterStatement> mList = new ArrayList<>();

		visited.put(is, mList);

		List<List<RewriterStatement>> mOptions = indices.stream().map(i -> generateSubtrees(stmt.getOperands().get(i), visited, ctx, maxCombinations)).collect(Collectors.toList());
		List<RewriterStatement> out = new ArrayList<>();

		for (int subsetMask = 0; subsetMask < totalSubsets; subsetMask++) {
			List<List<RewriterStatement>> mOptionCpy = new ArrayList<>(mOptions);

			for (int i = 0; i < n; i++) {
				// Check if the i-th child is included in the current subset
				if ((subsetMask & (1 << i)) == 0) {
					String dt = stmt.getOperands().get(indices.get(i)).getResultingDataType(ctx);
					String namePrefix = "tmp";
					if (dt.equals("MATRIX"))
						namePrefix = "M";
					else if (dt.equals("FLOAT"))
						namePrefix = "f";
					else if (dt.equals("INT"))
						namePrefix = "i";
					else if (dt.equals("BOOL"))
						namePrefix = "b";
					RewriterDataType mT = new RewriterDataType().as(namePrefix + rd.nextInt(100000)).ofType(dt);
					mT.consolidate(ctx);
					mOptionCpy.set(i, List.of(mT));
				}
			}

			out.addAll(mergeSubtreeCombinations(stmt, indices, mOptionCpy, ctx, maxCombinations));
			if (out.size() > maxCombinations) {
				System.out.println("Aborting early due to too many combinations");
				return out;
			}
		}

		return out;
	}

	public static final class Operand {
		public final String op;
		public final int numArgs;
		public final List<String>[] supportedTypes;
		public final boolean isLeaf;

		public Operand(String op, int numArgs, List<String>... supportedTypes) {
			this(op, numArgs, false, supportedTypes);
		}
		public Operand(String op, int numArgs, boolean isLeaf, List<String>... supportedTypes) {
			this.op = op;
			this.numArgs = numArgs;
			this.supportedTypes = supportedTypes;
			this.isLeaf = isLeaf;
		}

		public String toString() {
			return op;
		}
	}
}
