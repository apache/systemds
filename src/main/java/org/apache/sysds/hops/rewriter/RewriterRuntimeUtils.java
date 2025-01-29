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

package org.apache.sysds.hops.rewriter;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewriter.dml.DMLExecutor;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;

import javax.annotation.Nullable;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterRuntimeUtils {
	public static final boolean interceptAll = false;
	public static boolean printUnknowns = false;
	public static final String dbFile = "./src/test/resources/rewriterframework/expressions.db";
	public static final boolean readDB = true;
	public static final boolean writeDB = true;

	private static boolean setupComplete = false;

	private static HashMap<String, Integer> unknownOps = new HashMap<>();
	private static boolean ENFORCE_FLOAT_OBSERVATIONS = true; // To force every data type to float
	private static boolean OBSERVE_SELECTIONS = false;
	private static boolean OBSERVE_RAND = false;

	public static void setupIfNecessary() {
		if (setupComplete)
			return;

		setupComplete = true;

		if (interceptAll) {
			System.out.println("INTERCEPTOR");
			OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES = false;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = false;
			System.out.println("OptLevel:" + OptimizerUtils.getOptLevel().toString());
			System.out.println("AllowOpFusion: " + OptimizerUtils.ALLOW_OPERATOR_FUSION);
			System.out.println("AllowSumProductRewrites: " + OptimizerUtils.ALLOW_SUM_PRODUCT_REWRITES);
			System.out.println("AllowConstantFolding: " + OptimizerUtils.ALLOW_CONSTANT_FOLDING);

			// Setup default context
			RuleContext ctx = RewriterUtils.buildDefaultContext();

			RewriterDatabase exactExprDB = new RewriterDatabase();

			if (readDB) {
				try(BufferedReader reader = new BufferedReader(new FileReader(dbFile))) {
					exactExprDB.deserialize(reader, ctx);
				} catch (IOException ex) {
					ex.printStackTrace();
				}
			}

			RewriterRuntimeUtils.attachPreHopInterceptor(prog -> {
				RewriterRuntimeUtils.forAllUniqueTranslatableStatements(prog, 4, mstmt -> {}, exactExprDB, ctx);
				return true; // We will continue to extract the rewritten hop
			});

			RewriterRuntimeUtils.attachHopInterceptor(prog -> {
				RewriterRuntimeUtils.forAllUniqueTranslatableStatements(prog, 4, mstmt -> {}, exactExprDB, ctx);
				return false; // Then we cancel the excecution to save time
			});

			Runtime.getRuntime().addShutdownHook(new Thread(() -> {
				if (writeDB) {
					try (BufferedWriter writer = new BufferedWriter(new FileWriter(dbFile))) {
						exactExprDB.serialize(writer, ctx);
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}));
		}
	}

	public static void attachHopInterceptor(Function<DMLProgram, Boolean> interceptor) {
		DMLScript.hopInterceptor = interceptor;
	}

	public static void detachHopInterceptor() {
		DMLScript.hopInterceptor = null;
	}

	public static void attachPreHopInterceptor(Function<DMLProgram, Boolean> interceptor) {
		DMLScript.preHopInterceptor = interceptor;
	}

	public static void detachPreHopInterceptor() {
		DMLScript.preHopInterceptor = null;
	}

	public static RewriterStatement buildDAGFromHop(Hop hop, int maxDepth, boolean mindDataCharacteristics, final RuleContext ctx) {
		RewriterStatement out = buildDAGRecursively(hop, null, new HashMap<>(), 0, maxDepth, ctx);

		if (mindDataCharacteristics)
			return populateDataCharacteristics(out, ctx);

		return out;
	}

	public static RewriterStatement populateDataCharacteristics(RewriterStatement stmt, final RuleContext ctx) {
		if (stmt == null)
			return null;

		if (stmt instanceof RewriterDataType && stmt.getResultingDataType(ctx).equals("MATRIX")) {
			Long nrow = (Long) stmt.getMeta("_actualNRow");
			Long ncol = (Long) stmt.getMeta("_actualNCol");
			int matType = 0;

			if (nrow != null && nrow == 1L) {
				matType = 1;
			} else if (ncol != null && ncol == 1L) {
				matType = 2;
			}

			if (matType > 0) {
				return new RewriterInstruction()
						.as(stmt.getId())
						.withInstruction(matType == 1L ? "rowVec" : "colVec")
						.withOps(stmt)
						.consolidate(ctx);
			}
		}

		Map<RewriterStatement, RewriterStatement> createdObjects = new HashMap<>();

		stmt.forEachPostOrder((cur, pred) -> {
			for (int i = 0; i < cur.getOperands().size(); i++) {
				RewriterStatement child = cur.getChild(i);

				if (child instanceof RewriterDataType && child.getResultingDataType(ctx).equals("MATRIX")) {
					Long nrow = (Long) child.getMeta("_actualNRow");
					Long ncol = (Long) child.getMeta("_actualNCol");
					int matType = 0;

					if (nrow != null && nrow == 1L) {
						matType = 1;
					} else if (ncol != null && ncol == 1L) {
						matType = 2;
					}

					if (matType > 0) {
						RewriterStatement created = createdObjects.get(child);

						if (created == null) {
							created = new RewriterInstruction()
									.as(stmt.getId())
									.withInstruction(matType == 1 ? "rowVec" : "colVec")
									.withOps(child)
									.consolidate(ctx);
							createdObjects.put(child, created);
						}

						cur.getOperands().set(i, created);
					}
				}
			}
		}, false);

		return stmt;
	}

	public static void forAllUniqueTranslatableStatements(DMLProgram program, int maxDepth, Consumer<RewriterStatement> stmt, RewriterDatabase db, final RuleContext ctx) {
		try {
			Set<Hop> visited = new HashSet<>();

			for (String namespaceKey : program.getNamespaces().keySet()) {
				for (String fname : program.getFunctionStatementBlocks(namespaceKey).keySet()) {
					FunctionStatementBlock fsblock = program.getFunctionStatementBlock(namespaceKey, fname);
					handleStatementBlock(fsblock, maxDepth, stmt, visited, db, ctx);
				}
			}

			for (StatementBlock sb : program.getStatementBlocks()) {
				handleStatementBlock(sb, maxDepth, stmt, visited, db, ctx);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static void handleStatementBlock(StatementBlock sb, int maxDepth, Consumer<RewriterStatement> consumer, Set<Hop> visited, RewriterDatabase db, final RuleContext ctx) {
		if (sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
			fstmt.getBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			forAllUniqueTranslatableStatements(wsb.getPredicateHops(), maxDepth, consumer, visited, db, ctx);
			wstmt.getBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			forAllUniqueTranslatableStatements(isb.getPredicateHops(), maxDepth, consumer, visited, db, ctx);
			istmt.getIfBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
			istmt.getElseBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else if (sb instanceof ForStatementBlock)
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			forAllUniqueTranslatableStatements(fsb.getFromHops(), maxDepth, consumer, visited, db, ctx);
			forAllUniqueTranslatableStatements(fsb.getToHops(), maxDepth, consumer, visited, db, ctx);
			forAllUniqueTranslatableStatements(fsb.getIncrementHops(), maxDepth, consumer, visited, db, ctx);
			fstmt.getBody().forEach(s -> handleStatementBlock(s, maxDepth, consumer, visited, db, ctx));
		}
		else
		{
			if (sb.getHops() != null)
				sb.getHops().forEach(hop -> forAllUniqueTranslatableStatements(hop, maxDepth, consumer, visited, db, ctx));
		}
	}

	private static void forAllUniqueTranslatableStatements(Hop currentHop, int maxDepth, Consumer<RewriterStatement> consumer, Set<Hop> visited, RewriterDatabase db, final RuleContext ctx) {
		if (currentHop == null || visited.contains(currentHop))
			return;

		visited.add(currentHop);
		RewriterStatement stmt = buildDAGRecursively(currentHop, null, new HashMap<>(), 0, maxDepth, ctx);

		if (stmt instanceof RewriterInstruction)
			stmt = ctx.metaPropagator.apply(stmt);

		if (stmt == null) {
			// TODO: What to do about TWrite and PWrite?
			// Just ignore these ops?
			if (!currentHop.getOpString().startsWith("TWrite") && !currentHop.getOpString().startsWith("PWrite") && !currentHop.getValueType().toString().equals("STRING") && !currentHop.getOpString().startsWith("LiteralOp") && !currentHop.getOpString().startsWith("fcall") && !currentHop.getOpString().startsWith("TRead") && !currentHop.getOpString().startsWith("PRead"))
				unknownOps.compute(currentHop.getOpString() + "::" + currentHop.getDataType() + "::" + currentHop.getValueType(), (k, v) -> v == null ? 1 : v + 1);
		}

		if (stmt != null) {
			stmt.prepareForHashing();
			stmt.recomputeHashCodes(ctx);
		}

		if (stmt != null && db.insertEntry(ctx, stmt)) {
			RewriterStatement cpy = stmt.nestedCopyOrInject(new HashMap<>(), el -> null);
			consumer.accept(cpy);
		}

		if (currentHop.getInput() != null)
			currentHop.getInput().forEach(child -> forAllUniqueTranslatableStatements(child, maxDepth, consumer, visited, db, ctx));
	}

	private static RewriterStatement buildDAGRecursively(Hop next, @Nullable String expectedType, Map<Hop, RewriterStatement> cache, int depth, int maxDepth, final RuleContext ctx) {
		if (depth == maxDepth)
			return buildLeaf(next, expectedType, ctx);

		if (cache.containsKey(next))
			return checkForCorrectTypes(cache.get(next), expectedType, next, ctx);

		if (next instanceof LiteralOp) {
			RewriterStatement literal = buildLiteral((LiteralOp)next, expectedType, ctx);
			literal = checkForCorrectTypes(literal, expectedType, next, ctx);
			cache.put(next, literal);
			return literal;
		}

		if (next instanceof AggBinaryOp) {
			RewriterStatement stmt = buildAggBinaryOp((AggBinaryOp) next, expectedType, ctx);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof AggUnaryOp) {
			RewriterStatement stmt = buildAggUnaryOp((AggUnaryOp) next, expectedType, ctx);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof BinaryOp) {
			RewriterStatement stmt = buildBinaryOp((BinaryOp) next, expectedType, ctx);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof ReorgOp) {
			RewriterStatement stmt = buildReorgOp((ReorgOp) next, expectedType, ctx);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof UnaryOp) {
			RewriterStatement stmt = buildUnaryOp((UnaryOp)next, expectedType, ctx);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof IndexingOp) {
			RewriterStatement stmt = buildIndexingOp((IndexingOp) next, expectedType, ctx);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, next.getInput(), cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof DataGenOp) {
			List<Hop> interestingHops = new ArrayList<>();
			RewriterStatement stmt = buildDataGenOp((DataGenOp)next, expectedType, ctx, interestingHops);
			stmt = checkForCorrectTypes(stmt, expectedType, next, ctx);

			if (stmt == null)
				return buildLeaf(next, expectedType, ctx);

			insertDataCharacteristics(next, stmt, ctx);

			if (buildInputs(stmt, interestingHops, cache, true, depth, maxDepth, ctx))
				return stmt;

			return null;
		}

		if (next instanceof DataOp) {
			DataOp dop = (DataOp) next;

			if (dop.isRead())
				return buildLeaf(next, expectedType, ctx);
		}

		if (printUnknowns) {
			System.out.println("Unknown Op: " + next);
			System.out.println("Class: " + next.getClass().getSimpleName());
			System.out.println("OPString: " + next.getOpString());
		}

		return null;
	}

	private static void insertDataCharacteristics(Hop hop, RewriterStatement stmt, final RuleContext ctx) {
		if (stmt.getResultingDataType(ctx).equals("MATRIX")) {
			if (hop.getDataCharacteristics() != null) {
				long nrows = hop.getDataCharacteristics().getRows();
				long ncols = hop.getDataCharacteristics().getCols();
				if (nrows > 0)
					stmt.unsafePutMeta("_actualNRow", nrows);
				if (ncols > 0)
					stmt.unsafePutMeta("_actualNCol", ncols);
			}
		}
	}

	private static RewriterStatement checkForCorrectTypes(RewriterStatement stmt, @Nullable String expectedType, Hop hop, final RuleContext ctx) {
		if (stmt == null)
			return null;

		if (expectedType == null)
			expectedType = stmt.getResultingDataType(ctx);

		String actualType = resolveExactDataType(hop);

		if (actualType == null)
			return null;

		if (actualType.equals(expectedType))
			return stmt;

		if (actualType.equals("MATRIX")) {
			HashMap<String, RewriterStatement> oldTypes = new HashMap<>();
			oldTypes.put("A", stmt);
			RewriterStatement newStmt = RewriterUtils.parseExpression("as.matrix(A)", new HashMap<>(), oldTypes, ctx);
			return newStmt;
		}

		return null;
	}

	private static RewriterStatement buildLeaf(Hop hop, @Nullable String expectedType, final RuleContext ctx) {
		String hopName = hop.getName();

		// Check if hopName collides with literal values
		if (RewriterUtils.LONG_PATTERN.matcher(hopName).matches())
			hopName = "int" + new Random().nextInt(1000);
		if (RewriterUtils.DOUBLE_PATTERN.matcher(hopName).matches() || RewriterUtils.SPECIAL_FLOAT_PATTERN.matcher(hopName).matches())
			hopName = "float" + new Random().nextInt(1000);

		if (expectedType != null) {
			RewriterStatement stmt = RewriterUtils.parse(hopName, ctx, expectedType + ":" + hopName);
			insertDataCharacteristics(hop, stmt, ctx);
			return stmt;
		}

		switch (hop.getDataType()) {
			case SCALAR:
				return buildScalarLeaf(hop, hopName, ctx);
			case MATRIX:
				RewriterStatement stmt = RewriterUtils.parse(hopName, ctx, "MATRIX:" + hopName);
				insertDataCharacteristics(hop, stmt, ctx);
				return stmt;
		}

		return null; // Not supported then
	}

	private static RewriterStatement buildScalarLeaf(Hop hop, final RuleContext ctx) {
		return buildScalarLeaf(hop, null, ctx);
	}

	private static RewriterStatement buildScalarLeaf(Hop hop, @Nullable String newName, final RuleContext ctx) {
		if (newName == null)
			newName = hop.getName();

		switch (hop.getValueType()) {
			case FP64:
			case FP32:
				return RewriterUtils.parse(newName, ctx, "FLOAT:" + newName);
			case INT64:
			case INT32:
				if (ENFORCE_FLOAT_OBSERVATIONS)
					return RewriterUtils.parse(newName, ctx, "FLOAT:" + newName);
				return RewriterUtils.parse(newName, ctx, "INT:" + newName);
			case BOOLEAN:
				if (ENFORCE_FLOAT_OBSERVATIONS)
					return RewriterUtils.parse(newName, ctx, "FLOAT:" + newName);
				return RewriterUtils.parse(newName, ctx, "BOOL:" + newName);
		}

		return null; // Not supported then
	}

	private static boolean buildInputs(RewriterStatement stmt, List<Hop> inputs, Map<Hop, RewriterStatement> cache, boolean fixedSize, int depth, int maxDepth, final RuleContext ctx) {
		if (fixedSize && stmt.getOperands().size() != inputs.size())
			return false;

		List<RewriterStatement> children = new ArrayList<>();
		int ctr = 0;
		for (Hop in : inputs) {
			RewriterStatement childStmt = buildDAGRecursively(in, fixedSize ? stmt.getOperands().get(ctr).getResultingDataType(ctx) : null, cache, depth + 1, maxDepth, ctx);

			if (childStmt == null) {
				//System.out.println("Could not build child: " + in);
				// TODO: Then just build leaf
				//return false;
				childStmt = buildLeaf(in, stmt.getOperands().get(ctr).getResultingDataType(ctx), ctx);

				if (childStmt == null)
					return false;
			}

			if (fixedSize && !RewriterUtils.convertImplicitly(childStmt.getResultingDataType(ctx), ENFORCE_FLOAT_OBSERVATIONS).equals(stmt.getOperands().get(ctr).getResultingDataType(ctx)))
				throw new IllegalArgumentException("Different data type than expected: "  + stmt.toString(ctx) + "; [" + ctr + "] " + childStmt.toString(ctx) + " ::" + childStmt.getResultingDataType(ctx));

			children.add(childStmt);
			ctr++;
		}

		stmt.getOperands().clear();
		stmt.getOperands().addAll(children);
		stmt.consolidate(ctx);
		return true;
	}

	private static RewriterStatement buildIndexingOp(IndexingOp op, @Nullable String expectedType, final RuleContext ctx) {
		if (!OBSERVE_SELECTIONS)
			return null;

		if (expectedType == null) {
			expectedType = resolveExactDataType(op);

			if (expectedType == null)
				return null;
		}

		switch (op.getOpString()) {
			case "rix":
				return RewriterUtils.parse("[](A, i, j, k, l)", ctx, "MATRIX:A", "INT:i,j,k,l");
		}

		return null;
	}

	private static RewriterStatement buildUnaryOp(UnaryOp op, @Nullable String expectedType, final RuleContext ctx) {
		if (expectedType == null) {
			expectedType = resolveExactDataType(op);

			if (expectedType == null)
				return null;
		}

		String fromType = resolveExactDataType(op.getInput(0));
		Types.DataType toDT = op.getDataType();

		if (!toDT.isMatrix() && !toDT.isScalar())
			return null;

		switch(op.getOpString()) {
			case "u(castdts)":
				if (toDT.isMatrix())
					return RewriterUtils.parse("cast.MATRIX(A)", ctx, "MATRIX:A");
				if (fromType != null)
					return RewriterUtils.parse("cast." + expectedType + "(A)", ctx, fromType + ":A");

				return null;
			case "u(castdtm)":
				if (fromType != null)
					return RewriterUtils.parse("cast.MATRIX(a)", ctx, fromType + ":a");

				return null;
			case "u(sqrt)":
				return RewriterUtils.parse("sqrt(A)", ctx, fromType + ":A");
			case "u(!)":
				return RewriterUtils.parse("!(A)", ctx, fromType + ":A");
			case "u(ncol)":
				return RewriterUtils.parse("ncol(A)", ctx, "MATRIX:A");
			case "u(nrow)":
				return RewriterUtils.parse("nrow(A)", ctx, "MATRIX:A");
			case "u(length)":
				return RewriterUtils.parse("length(A)", ctx, "MATRIX:A");
			case "u(exp)":
				return RewriterUtils.parse("exp(A)", ctx, fromType + ":A");
			case "u(round)":
				return RewriterUtils.parse("round(A)", ctx, fromType + ":A");
			case "u(abs)":
				return RewriterUtils.parse("abs(A)", ctx, fromType + ":A");
		}

		if (printUnknowns)
			DMLExecutor.println("Unknown UnaryOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildAggBinaryOp(AggBinaryOp op, @Nullable String expectedType, final RuleContext ctx) {
		if (expectedType != null && !expectedType.equals("MATRIX"))
			throw new IllegalArgumentException();

		// Some placeholder definitions
		switch(op.getOpString()) {
			case "ba(+*)": // Matrix multiplication
				return RewriterUtils.parse("%*%(A, B)", ctx, "MATRIX:A,B");
		}

		if (printUnknowns)
			DMLExecutor.println("Unknown AggBinaryOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildAggUnaryOp(AggUnaryOp op, @Nullable String expectedType, final RuleContext ctx) {
		// Some placeholder definitions
		switch(op.getOpString()) {
			case "ua(+C)": // Matrix multiplication
				if (expectedType != null && !expectedType.equals("MATRIX"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return RewriterUtils.parse("colSums(A)", ctx, "MATRIX:A");
			case "ua(+R)":
				if (expectedType != null && !expectedType.equals("MATRIX"))
					throw new IllegalArgumentException("Unexpected type:" + expectedType);
				return RewriterUtils.parse("rowSums(A)", ctx, "MATRIX:A");
			case "ua(+RC)":
				if (expectedType != null && !expectedType.equals("FLOAT"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return RewriterUtils.parse("sum(A)", ctx, "MATRIX:A");
			case "ua(nrow)":
				if (expectedType != null && !expectedType.equals("INT"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return RewriterUtils.parse("nrow(A)", ctx, "MATRIX:A");
			case "ua(ncol)":
				if (expectedType != null && !expectedType.equals("INT"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return RewriterUtils.parse("ncol(A)", ctx, "MATRIX:A");
			case "ua(maxRC)":
				if (expectedType != null && !expectedType.equals("FLOAT"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return RewriterUtils.parse("max(A)", ctx, "MATRIX:A");
			case "ua(minRC)":
				if (expectedType != null && !expectedType.equals("FLOAT"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return RewriterUtils.parse("min(A)", ctx, "MATRIX:A");
			case "ua(traceRC)":
				return RewriterUtils.parse("trace(A)", ctx, "MATRIX:A");
		}

		if (printUnknowns)
			DMLExecutor.println("Unknown AggUnaryOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildBinaryOp(BinaryOp op, @Nullable String expectedType, final RuleContext ctx) {
		String t1 = resolveExactDataType(op.getInput().get(0));
		String t2 = resolveExactDataType(op.getInput().get(1));

		if (t1 == null || t2 == null)
			return null;

		t1 += ":a";
		t2 += ":b";

		RewriterStatement parsed = null;

		switch(op.getOpString()) {
			case "b(+)": // Addition
				parsed = RewriterUtils.parse("+(a, b)", ctx, t1, t2);
				break;
			case "b(*)": // Matrix multiplication
				parsed = RewriterUtils.parse("*(a, b)", ctx, t1, t2);
				break;
			case "b(-)":
				parsed = RewriterUtils.parse("-(a, b)", ctx, t1, t2);
				break;
			case "b(/)":
				parsed = RewriterUtils.parse("/(a, b)", ctx, t1, t2);
				break;
			case "b(||)":
				parsed = RewriterUtils.parse("|(a, b)", ctx, t1, t2);
				break;
			case "b(!=)":
				parsed = RewriterUtils.parse("!=(a, b)", ctx, t1, t2);
				break;
			case "b(==)":
				parsed = RewriterUtils.parse("==(a, b)", ctx, t1, t2);
				break;
			case "b(&&)":
				parsed = RewriterUtils.parse("&(a, b)", ctx, t1, t2);
				break;
			case "b(<)":
				parsed = RewriterUtils.parse("<(a, b)", ctx, t1, t2);
				break;
			case "b(>)":
				parsed = RewriterUtils.parse(">(a, b)", ctx, t1, t2);
				break;
			case "b(>=)":
				parsed = RewriterUtils.parse(">=(a, b)", ctx, t1, t2);
				break;
			case "b(<=)":
				parsed = RewriterUtils.parse("<=(a, b)", ctx, t1, t2);
				break;
			case "b(^)":
				parsed = RewriterUtils.parse("^(a, b)", ctx, t1, t2);
				break;
			case "b(rbind)":
				if (!t1.equals("MATRIX") || !t2.equals("MATRIX"))
					return null;
				return RewriterUtils.parse("RBind(a, b)", ctx, t1, t2);
			case "b(cbind)":
				if (!t1.equals("MATRIX") || !t2.equals("MATRIX"))
					return null;
				return RewriterUtils.parse("CBind(a, b)", ctx, t1, t2);
			case "b(1-*)":
				return RewriterUtils.parse("1-*(A, B)", ctx, "MATRIX:A,B");
		}

		if (parsed != null)
			return parsed.rename(op.getName());

		if (printUnknowns)
			DMLExecutor.println("Unknown BinaryOp: " + op.getOpString());
		return null;
	}

	private static String resolveExactDataType(Hop hop) {
		if (hop.getDataType() == Types.DataType.MATRIX)
			return "MATRIX";

		switch (hop.getValueType()) {
			case FP64:
			case FP32:
				return "FLOAT";
			case INT64:
			case INT32:
				if (ENFORCE_FLOAT_OBSERVATIONS)
					return "FLOAT";
				return "INT";
			case BOOLEAN:
				if (ENFORCE_FLOAT_OBSERVATIONS)
					return "FLOAT";
				return "BOOL";
		}

		if (printUnknowns)
			DMLExecutor.println("Unknown type: " + hop + " -> " + hop.getDataType() + " : " + hop.getValueType());

		return null;
	}

	private static RewriterStatement buildReorgOp(ReorgOp op, @Nullable String expectedType, final RuleContext ctx) {
		if (expectedType != null && !expectedType.equals("MATRIX"))
			throw new IllegalArgumentException();

		switch(op.getOpString()) {
			case "r(r')": // Matrix multiplication
				return RewriterUtils.parse("t(A)", ctx, "MATRIX:A");
			case "r(rev)":
				return RewriterUtils.parse("rev(A)", ctx, "MATRIX:A");
			case "r(rdiag)":
				return RewriterUtils.parse("diag(A)", ctx, "MATRIX:A");
		}

		//System.out.println("Unknown BinaryOp: " + op.getOpString());
		if (printUnknowns)
			DMLExecutor.println("Unknown ReorgOp: " + op.getOpString());
		return null;
	}

	private static RewriterStatement buildDataGenOp(DataGenOp op, @Nullable String expectedType, final RuleContext ctx, List<Hop> interestingHops) {
		if (expectedType != null && !expectedType.equals("MATRIX"))
			throw new IllegalArgumentException();

		switch(op.getOpString()) {
			case "dg(rand)":
				if (OBSERVE_RAND) {
					interestingHops.add(op.getParam("rows"));
					interestingHops.add(op.getParam("cols"));
					interestingHops.add(op.getParam("min"));
					interestingHops.add(op.getParam("max"));
					return RewriterUtils.parse("rand(i1, i2, f1, f2)", ctx, "INT:i1,i2", "FLOAT:f1,f2").rename(op.getName());
				}
				return null;
		}

		return null;
	}

	private static RewriterStatement buildLiteral(LiteralOp literal, @Nullable String expectedType, final RuleContext ctx) {
		if (literal.getDataType() != Types.DataType.SCALAR)
			return null; // Then it is not supported yet

		String mType;
		Object mValue;

		switch (literal.getValueType()) {
			case FP64:
			case FP32:
				if (expectedType != null && !expectedType.equals("FLOAT"))
					throw new IllegalArgumentException("Unexpected type: " + expectedType);
				return new RewriterDataType().as(UUID.randomUUID().toString()).ofType("FLOAT").asLiteral(literal.getDoubleValue()).consolidate(ctx);
			case INT32:
			case INT64:
				if (expectedType != null) {
					if (expectedType.equals("INT")) {
						mType = expectedType;
						mValue = literal.getLongValue();
					} else if (expectedType.equals("FLOAT")) {
						mType = "FLOAT";
						mValue = (double)literal.getLongValue();
					} else {
						throw new IllegalArgumentException();
					}
				} else {
					mType = "INT";
					mValue = literal.getLongValue();
				}
				return new RewriterDataType().as(UUID.randomUUID().toString()).ofType(mType).asLiteral(mValue).consolidate(ctx);
			case BOOLEAN:
				if (expectedType != null) {
					if (expectedType.equals("FLOAT")) {
						mType = expectedType;
						mValue = literal.getBooleanValue() ? 1.0D : 0.0D;
					} else if (expectedType.equals("INT")) {
						mType = expectedType;
						mValue = literal.getBooleanValue() ? 1L : 0L;
					} else if (expectedType.equals("BOOL")) {
						mType = expectedType;
						mValue = literal.getBooleanValue();
					} else {
						throw new IllegalArgumentException();
					}
				} else {
					mType = "BOOL";
					mValue = literal.getBooleanValue();
				}
				return new RewriterDataType().as(UUID.randomUUID().toString()).ofType(mType).asLiteral(mValue).consolidate(ctx);
			default:
				return null; // Not supported yet
		}
	}

	public static boolean executeScript(String script) {
		try {
			return DMLScript.executeScript(new String[]{"-s", script});
		} catch (Exception ex) {
			ex.printStackTrace();
			return false;
		}
	}


	/**
	 * Validates matrix dimensions to ensure that broadcasting still works afer the transformation
	 * @param hop1 the first HOP
	 * @param hop2 the second HOP
	 * @return if the new binary op would work in terms of broadcasting
	 */
	public static boolean validateBinaryBroadcasting(Hop hop1, Hop hop2) {
		if (hop1.isMatrix() && hop2.isMatrix()) {
			if (!hop1.dimsKnown() || !hop2.dimsKnown())
				return false;

			if (hop1.getDim1() == hop2.getDim1()) {
				if (hop1.getDim2() == hop2.getDim2())
					return true; // Then both dimensions match

				return hop2.getDim2() == 1; // Otherwise we require a column vector
			} else if (hop1.getDim2() == hop2.getDim2()) {
				return hop2.getDim1() == 1; // We require a row vector
			}

			// At least one dimension must match
			return false;
		}

		return true;
	}

	public static boolean hasMatchingDims(Hop hop1, Hop hop2) {
		return hop1.dimsKnown() && hop2.dimsKnown() && hop1.getDim1() == hop2.getDim1() && hop1.getDim2() == hop2.getDim2();
	}

	public static boolean hasMatchingDims(Hop... hops) {
		if (hops.length < 2)
			return true;

		for (Hop hop : hops)
			if (!hop.dimsKnown())
				return false;

		long dim1 = hops[0].getDim1();
		long dim2 = hops[0].getDim2();

		for (int i = 1; i < hops.length; i++)
			if (hops[i].getDim1() != dim1 && hops[i].getDim2() != dim2)
				return false;

		return true;
	}
}
