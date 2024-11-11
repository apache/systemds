package org.apache.sysds.hops.rewriter;

import org.apache.spark.internal.config.R;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.apache.sysds.hops.rewriter.RewriterContextSettings.ALL_TYPES;

public class RewriterRuleCollection {

	// Anything that can be substituted with 'a == b'
	public static void addEqualitySubstitutions(final List<RewriterRule> rules, final RuleContext ctx) {
		RewriterUtils.buildBinaryPermutations(List.of("MATRIX", "FLOAT", "INT", "BOOL"), (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.withParsedStatement("==(A,B)")
					.toParsedStatement("!(!=(A,B))")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.withParsedStatement("==(A,B)")
					.toParsedStatement("&(>=(A,B), <=(A,B))")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.withParsedStatement("==(A,B)")
					.toParsedStatement("!(&(>(A,B), <(A,B)))")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.parseGlobalVars("LITERAL_FLOAT:0")
					.withParsedStatement("==(A,B)")
					.toParsedStatement("==(+(A,-(B)),0)")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.parseGlobalVars("LITERAL_FLOAT:0")
					.withParsedStatement("==(A,B)")
					.toParsedStatement("==(+(-(A),B),0)")
					.build()
			);
		});

		ALL_TYPES.forEach(t -> {
			if (t.equals("MATRIX")) {
				rules.add(new RewriterRuleBuilder(ctx)
						.setUnidirectional(true)
						.parseGlobalVars(t + ":A")
						.parseGlobalVars("LITERAL_INT:1")
						.withParsedStatement("==(A,A)")
						.toParsedStatement("matrix(1, nrow(A), ncol(A))")
						.build()
				);

				rules.add(new RewriterRuleBuilder(ctx)
						.setUnidirectional(true)
						.parseGlobalVars("INT:r,c")
						.parseGlobalVars("LITERAL_INT:1")
						.withParsedStatement("matrix(1, r, c)")
						.toParsedStatement("==($1:_rdMATRIX(r, c),$1)")
						.build()
				);
			} else {
				rules.add(new RewriterRuleBuilder(ctx)
						.setUnidirectional(true)
						.parseGlobalVars(t + ":A")
						.parseGlobalVars("LITERAL_BOOL:TRUE")
						.withParsedStatement("==(A,A)")
						.toParsedStatement("TRUE")
						.build()
				);

				rules.add(new RewriterRuleBuilder(ctx)
						.setUnidirectional(true)
						.parseGlobalVars("LITERAL_BOOL:TRUE")
						.withParsedStatement("TRUE")
						.toParsedStatement("==($1:_rd" + t + "(),$1)")
						.build()
				);
			}
		});
	}

	public static void addBooleAxioms(final List<RewriterRule> rules, final RuleContext ctx) {
		// Identity axioms
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.withParsedStatement("a")
				.toParsedStatement("|(a, FALSE)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.withParsedStatement("a")
				.toParsedStatement("&(a, TRUE)")
				.build()
		);

		// Domination axioms
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.withParsedStatement("TRUE")
				.toParsedStatement("|(_anyBool(), TRUE)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.withParsedStatement("|(a, TRUE)")
				.toParsedStatement("TRUE")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.withParsedStatement("FALSE")
				.toParsedStatement("&(_anyBool(), FALSE)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.withParsedStatement("&(a, FALSE)")
				.toParsedStatement("FALSE")
				.build()
		);

		// Idempotence axioms
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a")
				.withParsedStatement("a")
				.toParsedStatement("|(a, a)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a")
				.withParsedStatement("a")
				.toParsedStatement("&(a, a)")
				.build()
		);

		// Commutativity
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a,b")
				.withParsedStatement("|(a, b)")
				.toParsedStatement("|(b, a)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a,b")
				.withParsedStatement("&(a, b)")
				.toParsedStatement("&(b, a)")
				.build()
		);

		// Associativity
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a,b,c")
				.withParsedStatement("|(|(a, b), c)")
				.toParsedStatement("|(a, |(b, c))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a,b,c")
				.withParsedStatement("&(&(a, b), c)")
				.toParsedStatement("&(a, &(b, c))")
				.build()
		);

		// Distributivity
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a,b,c")
				.withParsedStatement("&(a, |(b, c))")
				.toParsedStatement("|(&(a, b), &(a, c))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a,b,c")
				.withParsedStatement("&(&(a, b), c)")
				.toParsedStatement("&(a, &(b, c))")
				.build()
		);

		// Complementation
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.withParsedStatement("TRUE")
				.toParsedStatement("|($1:_anyBool(), !($1))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.withParsedStatement("|(a, !(a))")
				.toParsedStatement("TRUE")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.withParsedStatement("FALSE")
				.toParsedStatement("&($1:_anyBool(), !($1))")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("BOOL:a")
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.withParsedStatement("&(a, !(a))")
				.toParsedStatement("FALSE")
				.build()
		);

		// Double negation
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("BOOL:a")
				.withParsedStatement("a")
				.toParsedStatement("!(!(a))")
				.build()
		);


		/*RewriterUtils.buildBinaryPermutations(List.of("MATRIX", "FLOAT", "INT", "BOOL"), (t1, t2) -> {
			boolean isBool = t1.equals("BOOL") && t2.equals("BOOL");
			// Identity axioms
			rules.add(new RewriterRuleBuilder(ctx)
					.parseGlobalVars(t1 + ":A")
					.parseGlobalVars(t2 + ":B")
					.parseGlobalVars("LITERAL_FLOAT:0")
					.withParsedStatement("!=(A,0)")
					.toParsedStatement("!(!=(A,B))")
					.build()
			);
		});*/
	}

	public static void addImplicitBoolLiterals(final List<RewriterRule> rules, final RuleContext ctx) {
		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("TRUE")
				.toParsedStatement("<(_lower(1), 1)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("LITERAL_BOOL:TRUE")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("TRUE")
				.toParsedStatement(">(_higher(1), 1)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("FALSE")
				.toParsedStatement("<(_higher(1), 1)")
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.parseGlobalVars("LITERAL_BOOL:FALSE")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("FALSE")
				.toParsedStatement(">(_lower(1), 1)")
				.build()
		);
	}

	public static RewriterHeuristic getHeur(final RuleContext ctx) {
		ArrayList<RewriterRule> preparationRules = new ArrayList<>();

		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (t1, t2) -> {
			Stream.of("&", "|").forEach(expr -> {
				preparationRules.add(new RewriterRuleBuilder(ctx)
						.setUnidirectional(true)
						.parseGlobalVars(t1 + ":a")
						.parseGlobalVars(t2 + ":b")
						.withParsedStatement(expr + "(a, b)")
						.toParsedStatement(expr + "(_asVar(a), b)")
						.iff(match -> match.getMatchRoot().getOperands().get(0).isLiteral()
								|| (match.getMatchRoot().getOperands().get(0).isInstruction()
								&& match.getMatchRoot().getOperands().get(0).trueInstruction().startsWith("_")
								&& !match.getMatchRoot().getOperands().get(0).trueInstruction().equals("_asVar")), true)
						.build()
				);
				preparationRules.add(new RewriterRuleBuilder(ctx)
						.setUnidirectional(true)
						.parseGlobalVars(t1 + ":a")
						.parseGlobalVars(t2 + ":b")
						.withParsedStatement(expr + "(a, b)")
						.toParsedStatement(expr + "(a, _asVar(b))")
						.iff(match -> match.getMatchRoot().getOperands().get(1).isLiteral()
								|| (match.getMatchRoot().getOperands().get(1).isInstruction()
								&& match.getMatchRoot().getOperands().get(1).trueInstruction().startsWith("_")
								&& !match.getMatchRoot().getOperands().get(1).trueInstruction().equals("_asVar")), true)
						.build()
				);
			});
		});

		ALL_TYPES.forEach(t -> preparationRules.add((new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars(t + ":a")
				.withParsedStatement("!(a)")
				.toParsedStatement("!(_asVar(a))")
				.iff(match -> match.getMatchRoot().getOperands().get(0).isLiteral()
						|| (match.getMatchRoot().getOperands().get(0).isInstruction()
						&& match.getMatchRoot().getOperands().get(0).trueInstruction().startsWith("_")
						&& !match.getMatchRoot().getOperands().get(0).trueInstruction().equals("_asVar")), true)
				.build()
		)));

		RewriterRuleSet rs = new RewriterRuleSet(ctx, preparationRules);
		rs.accelerate();

		return new RewriterHeuristic(rs, true);
	}

	public static void canonicalizeAlgebraicStatements(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx, "-(a,b) => +(a,-(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("-(a, b)", hooks)
					.toParsedStatement("+(a, -(b))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "/(a,b) => *(a, inv(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("/(a, b)", hooks)
					.toParsedStatement("*(a, inv(b))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "-(+(a, b)) => +(-(a), -(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("-(+(a, b))", hooks)
					.toParsedStatement("$1:+(-(a), -(b))", hooks)
					/*.iff(match -> {System.out.println("Parent: " + match.getPredecessor().getParent()); System.out.println("Is Meta: " + match.getPredecessor().isMetaObject()); System.out.println("Child: " + match.getMatchRoot()); return true;}, true)
							.apply(hooks.get(1).getId(), (t, match) -> {System.out.println("New: " + t); System.out.println("New Assertions: " + match.getNewExprRoot().getAssertions(ctx));}, true)*/
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "-(-(a)) => a")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("-(-(a))", hooks)
					.toParsedStatement("a", hooks)
					.build()
			);
		});

		rules.add(new RewriterRuleBuilder(ctx, "length(A) => nrow(A) * ncol(A)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.withParsedStatement("length(A)", hooks)
				.toParsedStatement("*(nrow(A), ncol(A))", hooks)
				.build()
		);
	}

	public static void canonicalizeBooleanStatements(final List<RewriterRule> rules, final RuleContext ctx) {
		// TODO: Constant folding, but maybe not as successive rules
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		RewriterUtils.buildBinaryPermutations(ALL_TYPES, (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx, ">(a, b) => <(b, a)")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement(">(a, b)", hooks)
					.toParsedStatement("<(b, a)", hooks)
					.build()
			);

			// These hold only for boolean expressions
			/*rules.add(new RewriterRuleBuilder(ctx, "!(!(a)) = a")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("!(!(a))", hooks)
					.toParsedStatement("a", hooks)
					.build()
			);*/

			rules.add(new RewriterRuleBuilder(ctx, "<=(a, b) => |(<(a, b), ==(a, b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("<=(a, b)", hooks)
					.toParsedStatement("|(<(a, b), ==(a, b))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, ">=(a, b) => |(<(b, a), ==(b, a))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement(">=(a, b)", hooks)
					.toParsedStatement("|(<(b, a), ==(b, a))", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "!(&(a, b)) => |(!(a), !(b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("!(&(a, b))", hooks)
					.toParsedStatement("|(!(a), !(b))", hooks)
					.build()
			);

			List.of("&(a, b)", "&(b, a)").forEach(exp -> {
				List.of("|(" + exp + ", a)", "|(a, " + exp + ")").forEach(tExpr -> {
					rules.add(new RewriterRuleBuilder(ctx, tExpr + " => a")
							.setUnidirectional(true)
							.parseGlobalVars(t1 + ":a")
							.parseGlobalVars(t2 + ":b")
							.withParsedStatement(tExpr, hooks)
							.toParsedStatement("a", hooks)
							.build()
					);
				});
			});

			List.of("|(a, b)", "|(b, a)").forEach(exp -> {
				List.of("&(" + exp + ", a)", "&(a, " + exp + ")").forEach(tExpr -> {
					rules.add(new RewriterRuleBuilder(ctx, tExpr + " => a")
							.setUnidirectional(true)
							.parseGlobalVars(t1 + ":a")
							.parseGlobalVars(t2 + ":b")
							.withParsedStatement(tExpr, hooks)
							.toParsedStatement("a", hooks)
							.build()
					);
				});
			});

			rules.add(new RewriterRuleBuilder(ctx,  "|(<(b, a), <(a, b)) => b != a")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.withParsedStatement("|(<(b, a), <(a, b))", hooks)
					.toParsedStatement("!=(b, a)", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx,  "&(<(b, a), <(a, b)) => FALSE")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars("LITERAL_BOOL:FALSE")
					.withParsedStatement("&(<(b, a), <(a, b))", hooks)
					.toParsedStatement("FALSE", hooks)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx,  "!(!=(a, b)) => ==(a, b)")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars("LITERAL_BOOL:FALSE")
					.withParsedStatement("!(!=(a, b))", hooks)
					.toParsedStatement("==(a, b)", hooks)
					.build()
			);

			// TODO: Introduce e-class
			/*rules.add(new RewriterRuleBuilder(ctx,  "==(a, b) => isZero(+(a, -(b)))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars("LITERAL_BOOL:FALSE")
					.withParsedStatement("!(!=(a, b))", hooks)
					.toParsedStatement("==(a, b)", hooks)
					.build()
			);*/
		});
	}

	// E.g. expand A * B -> _m($1:_idx(), 1, nrow(A), _m($2:_idx(), 1, nrow(B), A[$1, $2] * B[$1, $2]))
	public static void expandStreamingExpressions(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();


		// Matrix Multiplication
		rules.add(new RewriterRuleBuilder(ctx, "Expand matrix product")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("%*%(A, B)", hooks)
				.toParsedStatement("$4:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(B)), sum($5:_m($3:_idx(1, ncol(A)), 1, *([](A, $1, $3), [](B, $3, $2)))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(3).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(4).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);

					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					RewriterStatement bRef = stmt.getChild(1, 1, 0);
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNCol(), bRef.getNRow());
				}, true) // Assumes it will never collide
				.apply(hooks.get(5).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true) // Assumes it will never collide
				.build()
		);

		// E.g. A + B
		rules.add(new RewriterRuleBuilder(ctx, "Expand Element Wise Instruction")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("$1:ElementWiseInstruction(A,B)", hooks)
				.toParsedStatement("$7:_m($2:_idx(1, $5:nrow(A)), $3:_idx(1, $6:ncol(A)), $4:ElementWiseInstruction([](A, $2, $3), [](B, $2, $3)))", hooks)
				/*.iff(match -> {
					return !match.getMatchRoot().isInstruction() || match.getMatchRoot().trueInstruction().equals("_m");
				}, true)*/
				.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(3).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(7).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);

					// Now we assert that nrow(A) = nrow(B) and ncol(A) = ncol(B)
					RewriterStatement aRef = stmt.getChild(2, 0, 0);
					RewriterStatement bRef = stmt.getChild(2, 1, 0);
					/*System.out.println("aNRow: " + aRef.getNRow());
					System.out.println("bNRow: " + bRef.getNRow());
					System.out.println("HERE1: " + match.getNewExprRoot().toParsableString(ctx));*/
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNRow(), bRef.getNRow());
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNCol(), bRef.getNCol());
					/*System.out.println(match.getNewExprRoot().getAssertions(ctx).getAssertions(aRef.getNRow()));
					System.out.println(match.getMatchRoot());
					System.out.println("HERE2: " + match.getNewExprRoot().toParsableString(ctx));*/
				}, true) // Assumes it will never collide
				//.apply(hooks.get(5).getId(), stmt -> stmt.unsafePutMeta("dontExpand", true), true)
				//.apply(hooks.get(6).getId(), stmt -> stmt.unsafePutMeta("dontExpand", true), true)
				.build()
		);

		List.of("$2:_m(i, j, v1), v2", "v1, $2:_m(i, j, v2)").forEach(s -> {
			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A,B")
					.parseGlobalVars("LITERAL_INT:1")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v1,v2")
					.withParsedStatement("$1:ElementWiseInstruction(" + s + ")", hooks)
					.toParsedStatement("$3:_m(i, j, $4:ElementWiseInstruction(v1, v2))", hooks)
					.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
					.build()
			);
		});

		// Trace(A)
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("trace(A)", hooks)
				.toParsedStatement("sum($3:_m($1:_idx(1, $2:nrow(A)), 1, [](A, $1, $1)))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("dontExpand", true), true)
				.apply(hooks.get(3).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					//stmt.getOperands().get(1).unsafePutMeta("ownerId", id);

					// Assert that the matrix is squared
					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					/*System.out.println("NewRoot: " + match.getNewExprRoot());
					System.out.println("aRef: " + aRef);
					System.out.println("nRow: " + aRef.getNRow());
					System.out.println("nCol: " + aRef.getNCol());*/
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNRow(), aRef.getNCol());
				}, true)
				.build()
		);

		// t(A)
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("t(A)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, ncol(A)), $2:_idx(1, nrow(A)), [](A, $2, $1))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// rand(rows, cols, min, max)
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.parseGlobalVars("INT:n,m")
				.parseGlobalVars("FLOAT:a,b")
				.withParsedStatement("rand(n, m, a, b)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, n), $2:_idx(1, m), +(a, *(+(b, -(a)), rand($1, $2))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// sum(A) = sum(_m($1:_idx(1, nrow(A)), 1, sum(_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2)))))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("sum(A)", hooks)
				.toParsedStatement("sum($3:_m($1:_idx(1, nrow(A)), 1, sum($4:_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2)))))", hooks)
				.iff(match -> {
					RewriterStatement meta = (RewriterStatement) match.getMatchRoot().getOperands().get(0).getMeta("ncol");

					if (meta == null)
						throw new IllegalArgumentException("Column meta should not be null: " + match.getMatchRoot().getOperands().get(0).toString(ctx));

					return !meta.isLiteral() || ((long)meta.getLiteral()) != 1;
				}, true)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// rowSums(A) -> _m($1:_idx(1, nrow(A)), 1, sum(_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2)))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("rowSums(A)", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, nrow(A)), 1, sum($4:_m($2:_idx(1, ncol(A)), 1, [](A, $1, $2))))", hooks)
				.iff(match -> {
					RewriterStatement meta = (RewriterStatement) match.getMatchRoot().getOperands().get(0).getMeta("ncol");

					if (meta == null)
						throw new IllegalArgumentException("Column meta should not be null: " + match.getMatchRoot().getOperands().get(0).toString(ctx));

					return !meta.isLiteral() || ((long)meta.getLiteral()) != 1;
				}, true)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// colSums(A) -> _m($1:_idx(1, ncol(A)), 1, sum(_m($2:_idx(1, nrow(A)), 1, [](A, $2, $1)))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("colSums(A)", hooks)
				.toParsedStatement("$3:_m(1, $1:_idx(1, ncol(A)), sum($4:_m($2:_idx(1, nrow(A)), 1, [](A, $2, $1))))", hooks)
				.iff(match -> {
					RewriterStatement meta = (RewriterStatement) match.getMatchRoot().getOperands().get(0).getMeta("ncol");

					if (meta == null)
						throw new IllegalArgumentException("Column meta should not be null: " + match.getMatchRoot().getOperands().get(0).toString(ctx));

					return !meta.isLiteral() || ((long)meta.getLiteral()) != 1;
				}, true)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("_idx(1, 1)", hooks)
				.toParsedStatement("$1:1", hooks)
				.build()
		);

		// TODO: Continue
		// Scalars dependent on matrix to index streams
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("sum(A)", hooks)
				.toParsedStatement("sum($3:_idxExpr($1:_idx(1, nrow(A)), $4:_idxExpr($2:_idx(1, ncol(A)), [](A, $1, $2))))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
				}, true)
				.build()
		);

		// TODO: Handle nrow / ncol equivalence (maybe need some kind of E-Graph after all)
		// diag(A) -> _m($1:_idx(1, nrow(A)), 1, [](A, $1, $1))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("diag(A)", hooks)
				.toParsedStatement("$2:_m($1:_idx(1, nrow(A)), 1, [](A, $1, $1))", hooks)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), (stmt, match) -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);

					RewriterStatement aRef = stmt.getChild(0, 1, 0);
					match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(aRef.getNRow(), aRef.getNCol());
				}, true)
				.build()
		);

		// cast.MATRIX(a) => _m(1, 1, a)
		for (String t : List.of("INT", "BOOL", "FLOAT")) {
			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars(t + ":a")
					.parseGlobalVars("LITERAL_INT:1")
					.withParsedStatement("cast.MATRIX(a)", hooks)
					.toParsedStatement("$2:_m(1, 1, a)", hooks)
					.apply(hooks.get(2).getId(), (stmt, match) -> stmt.unsafePutMeta("ownerId", UUID.randomUUID()), true)
					.build()
			);
		}
	}

	public static void expandArbitraryMatrices(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();
		// This must be the last rule in the heuristic as it handles any matrix that has not been written as a stream
		// A -> _m()
		rules.add(new RewriterRuleBuilder(ctx, "Expand arbitrary matrix expression")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("A", hooks)
				.toParsedStatement("$3:_m($1:_idx(1, nrow(A)), $2:_idx(1, ncol(A)), [](A, $1, $2))", hooks)
				.iff(match -> {
					// TODO: Does not work like this bc cyclic references
					/*RewriterStatement root = match.getMatchRoot();
					RewriterStatement parent = match.getMatchParent();
					// TODO: This check has to be extended to any meta expression
					return !(root.isInstruction() && root.trueInstruction().equals("_m"))
							&& (parent == null || (!parent.trueInstruction().equals("[]") && !parent.trueInstruction().equals("ncol") && !parent.trueInstruction().equals("nrow")));*/
					//System.out.println("HERE");
					return match.getMatchRoot().getMeta("dontExpand") == null && !(match.getMatchRoot().isInstruction() && match.getMatchRoot().trueInstruction().equals("_m"));
				}, true)
				.apply(hooks.get(1).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true) // Assumes it will never collide
				.apply(hooks.get(2).getId(), stmt -> stmt.unsafePutMeta("idxId", UUID.randomUUID()), true)
				.apply(hooks.get(3).getId(), stmt -> {
					UUID id = UUID.randomUUID();
					stmt.unsafePutMeta("ownerId", id);
					stmt.getOperands().get(0).unsafePutMeta("ownerId", id);
					stmt.getOperands().get(1).unsafePutMeta("ownerId", id);
					RewriterStatement A = stmt.getChild(0, 1, 0);
					A.unsafePutMeta("dontExpand", true);
					// TODO:
					//System.out.println("A: " + A);
					//System.out.println("ncol: " + A.getNCol());
					if (A.getNRow().isInstruction() && A.getNRow().trueInstruction().equals("nrow") && A.getNRow().getChild(0) == stmt)
						A.getNRow().getOperands().set(0, A);
					if (A.getNCol().isInstruction() && A.getNCol().trueInstruction().equals("ncol") && A.getNCol().getChild(0) == stmt)
						A.getNCol().getOperands().set(0, A);
					//System.out.println("newNRow: " + A.getNRow());
				}, true)
				.build()
		);
	}

	// TODO: Big issue when having multiple references to the same sub-dag
	public static void pushdownStreamSelections(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		rules.add(new RewriterRuleBuilder(ctx, "Element selection pushdown")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:h,i,j,k,l,m")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("[]($1:_m(h, i, v), l, m)", hooks)
				.toParsedStatement("as.scalar($2:_m(l, m, v))", hooks)
				/*.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands().get(0).getOperands();
					return ops.get(0).isInstruction()
							&& ops.get(1).isInstruction()
							&& ops.get(0).trueTypedInstruction(ctx).equals("_idx(INT,INT)")
							&& ops.get(1).trueTypedInstruction(ctx).equals("_idx(INT,INT)");
				}, true)*/
				.linkUnidirectional(hooks.get(1).getId(), hooks.get(2).getId(), lnk -> {
					RewriterStatement.transferMeta(lnk);
					/*UUID ownerId = (UUID)lnk.newStmt.get(0).getMeta("ownerId");
					System.out.println("OwnerId: " + ownerId);
					lnk.newStmt.get(0).getOperands().get(0).unsafePutMeta("ownerId", ownerId);
					lnk.newStmt.get(0).getOperands().get(0).unsafePutMeta("idxId", UUID.randomUUID());
					lnk.newStmt.get(0).getOperands().get(1).unsafePutMeta("ownerId", ownerId);
					lnk.newStmt.get(0).getOperands().get(1).unsafePutMeta("idxId", UUID.randomUUID());*/

					// TODO: Big issue when having multiple references to the same sub-dag
					for (int idx = 0; idx < 2; idx++) {
						RewriterStatement oldRef = lnk.oldStmt.getOperands().get(idx);
						RewriterStatement newRef = lnk.newStmt.get(0).getOperands().get(idx);

						// Replace all references to h with
						lnk.newStmt.get(0).getOperands().get(2).forEachPreOrder((el, pred) -> {
							for (int i = 0; i < el.getOperands().size(); i++) {
								RewriterStatement child = el.getOperands().get(i);
								Object meta = child.getMeta("idxId");

								if (meta instanceof UUID && meta.equals(oldRef.getMeta("idxId")))
									el.getOperands().set(i, newRef);
							}
							return true;
						}, false);

					}
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "Selection pushdown")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:h,i,j,k,l,m")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("[]($1:_m(h, i, v), j, k, l, m)", hooks)
				.toParsedStatement("$2:_m(_idx(j, l), _idx(k, m), v)", hooks) // Assuming that selections are valid
				/*.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands().get(0).getOperands();
					return ops.get(0).isInstruction()
							&& ops.get(1).isInstruction()
							&& ops.get(0).trueTypedInstruction(ctx).equals("_idx(INT,INT)")
							&& ops.get(1).trueTypedInstruction(ctx).equals("_idx(INT,INT)");
				}, true)*/
				.linkUnidirectional(hooks.get(1).getId(), hooks.get(2).getId(), lnk -> {
					// TODO: Big issue when having multiple references to the same sub-dag
					// BUT: This should usually not happen if indices are never referenced
					RewriterStatement.transferMeta(lnk);
					/*UUID ownerId = (UUID)lnk.newStmt.get(0).getMeta("ownerId");
					lnk.newStmt.get(0).getOperands().get(0).unsafePutMeta("ownerId", ownerId);
					lnk.newStmt.get(0).getOperands().get(0).unsafePutMeta("idxId", UUID.randomUUID());
					lnk.newStmt.get(0).getOperands().get(1).unsafePutMeta("ownerId", ownerId);
					lnk.newStmt.get(0).getOperands().get(1).unsafePutMeta("idxId", UUID.randomUUID());*/

					//if (ownerId == null)
						//throw new IllegalArgumentException();

					for (int idx = 0; idx < 2; idx++) {
						RewriterStatement oldRef = lnk.oldStmt.getOperands().get(idx);
						RewriterStatement newRef = lnk.newStmt.get(0).getOperands().get(idx);

						// Replace all references to h with
						lnk.newStmt.get(0).getOperands().get(2).forEachPreOrder((el, pred) -> {
							for (int i = 0; i < el.getOperands().size(); i++) {
								RewriterStatement child = el.getOperands().get(i);
								Object meta = child.getMeta("idxId");

								if (meta instanceof UUID && meta.equals(oldRef.getMeta("idxId")))
									el.getOperands().set(i, newRef);
							}
							return true;
						}, false);

					}
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "Eliminate scalar matrices")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("as.scalar(_m(i, j, v))", hooks)
				.toParsedStatement("v", hooks)
				.build()
		);

		// TODO: Deal with boolean or int matrices
		rules.add(new RewriterRuleBuilder(ctx, "_m(i::<const>, j::<const>, v) => cast.MATRIX(v)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_m(i, j, v)", hooks)
				.toParsedStatement("cast.MATRIX(v)", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands();

					boolean matching = (!ops.get(0).isInstruction() || !ops.get(0).trueInstruction().equals("_idx") || ops.get(0).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"))
							&& (!ops.get(1).isInstruction() || !ops.get(1).trueInstruction().equals("_idx") || ops.get(1).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"));

					return matching;
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idx(a,a) => a")
				.setUnidirectional(true)
				.parseGlobalVars("INT:a")
				.withParsedStatement("_idx(a,a)", hooks)
				.toParsedStatement("a", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i::<const>, v) => v")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_idxExpr(i, v)", hooks)
				.toParsedStatement("v", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands();

					boolean matching = (!ops.get(0).isInstruction() || !ops.get(0).trueInstruction().equals("_idx") || ops.get(0).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"))
							&& (!ops.get(1).isInstruction() || !ops.get(1).trueInstruction().equals("_idx") || ops.get(1).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"));

					return matching;
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i::<const>, v) => v")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("_idxExpr(i, v)", hooks)
				.toParsedStatement("v", hooks)
				.iff(match -> {
					List<RewriterStatement> ops = match.getMatchRoot().getOperands();

					boolean matching = (!ops.get(0).isInstruction() || !ops.get(0).trueInstruction().equals("_idx") || ops.get(0).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"))
							&& (!ops.get(1).isInstruction() || !ops.get(1).trueInstruction().equals("_idx") || ops.get(1).getMeta("ownerId") != match.getMatchRoot().getMeta("ownerId"));

					return matching;
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i, sum(...)) => sum(_idxExpr(i, ...))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("$1:_idxExpr(i, sum(v))", hooks)
				.toParsedStatement("sum($2:_idxExpr(i, v))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "_idxExpr(i, sum(...)) => sum(_idxExpr(i, ...))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("$1:_idxExpr(i, sum(v))", hooks)
				.toParsedStatement("sum($2:_idxExpr(i, v))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		RewriterUtils.buildBinaryPermutations(List.of("FLOAT"), (t1, t2) -> {
			// TODO: This probably first requires pulling out invariants of this idxExpr
			rules.add(new RewriterRuleBuilder(ctx, "*(sum(_idxExpr(i, ...)), sum(_idxExpr(j, ...))) => _idxExpr(i, _idxExpr(j, sum(*(...)))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A,B")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars(t1 + ":v1")
					.parseGlobalVars(t2 + ":v2")
					.withParsedStatement("$1:*(sum($2:_idxExpr(i, v1)), sum($3:_idxExpr(j, v2)))", hooks)
					.toParsedStatement("sum($4:_idxExpr(i, $5:_idxExpr(j, $6:*(v1, v2))))", hooks)
					.link(hooks.get(1).getId(), hooks.get(6).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(2).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(3).getId(), hooks.get(5).getId(), RewriterStatement::transferMeta)
					.build()
			);
		});

		rules.add(new RewriterRuleBuilder(ctx, "sum(sum(v)) => sum(v)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("sum(sum(v))", hooks)
				.toParsedStatement("sum(v)", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sum(sum(v)) => sum(v)")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT*:v")
				.withParsedStatement("sum(sum(v))", hooks)
				.toParsedStatement("sum(v)", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "[](UnaryElementWiseOperator(A), i, j) => UnaryElementWiseOperator([](A, i, j))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT:i,j")
				.withParsedStatement("[]($1:UnaryElementWiseOperator(A), i, j)", hooks)
				.toParsedStatement("$2:UnaryElementWiseOperator([](A, i, j))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "[](ElementWiseUnary.FLOAT(A), i, j) => ElementWiseUnary.FLOAT([](A, i, j))")
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A")
				.parseGlobalVars("INT:i,j")
				.withParsedStatement("[]($1:ElementWiseUnary.FLOAT(A), i, j)", hooks)
				.toParsedStatement("$2:ElementWiseUnary.FLOAT([](A, i, j))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		for (String t : ALL_TYPES) {
			if (t.equals("MATRIX")) {
				rules.add(new RewriterRuleBuilder(ctx, "ElementWiseInstruction(_m(i, j, v), b) => _m(i, j, ElementWiseInstruction(v, b))")
						.setUnidirectional(true)
						.parseGlobalVars("FLOAT:v")
						.parseGlobalVars(t + ":B")
						.parseGlobalVars("INT:i,j")
						.withParsedStatement("$1:ElementWiseInstruction($2:_m(i, j, v), B)", hooks)
						.toParsedStatement("$3:_m(i, j, $4:ElementWiseInstruction(v, [](B, i, j)))", hooks)
						.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
						.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
						.apply(hooks.get(3).getId(), (stmt, match) -> {
							// Then we an infer that the two matrices have the same dimensions
							match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(stmt.getNCol(), stmt.getChild(2, 1, 0).getNCol());
							match.getNewExprRoot().getAssertions(ctx).addEqualityAssertion(stmt.getNRow(), stmt.getChild(2, 1, 0).getNRow());
						}, true)
						.build()
				);

				continue;
			}
			rules.add(new RewriterRuleBuilder(ctx, "ElementWiseInstruction(_m(i, j, A), b) => _m(i, j, ElementWiseInstruction(A, b))")
					.setUnidirectional(true)
					.parseGlobalVars("FLOAT:v")
					.parseGlobalVars(t + ":b")
					.parseGlobalVars("INT:i,j")
					.withParsedStatement("$1:ElementWiseInstruction($2:_m(i, j, v), b)", hooks)
					.toParsedStatement("$3:_m(i, j, $4:ElementWiseInstruction(v, b))", hooks)
					.link(hooks.get(1).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "[](ElementWiseInstruction(A, v), i, j) => ElementWiseInstruction(v, [](A, i, j))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":v")
					.parseGlobalVars("INT:i,j")
					.withParsedStatement("[]($1:ElementWiseInstruction(A, v), i, j)", hooks)
					.toParsedStatement("$2:ElementWiseInstruction([](A, i, j), v)", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "[](ElementWiseInstruction(v, A), i, j) => ElementWiseInstruction(v, [](A, i, j))")
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":v")
					.parseGlobalVars("INT:i,j")
					.withParsedStatement("[]($1:ElementWiseInstruction(v, A), i, j)", hooks)
					.toParsedStatement("$2:ElementWiseInstruction(v, [](A, i, j))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.build()
			);
		}
	}

	// This expands the statements to a common canonical form
	// It is important, however, that
	public static void canonicalExpandAfterFlattening(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		rules.add(new RewriterRuleBuilder(ctx, "sum($1:_idxExpr(indices, -(A))) => -(sum($2:_idxExpr(indices, A)))")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT:a")
				.parseGlobalVars("INT...:indices")
				.withParsedStatement("sum($1:_idxExpr(indices, -(a)))", hooks)
				.toParsedStatement("-(sum($2:_idxExpr(indices, a)))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sum($1:_idxExpr(indices, -(a))) => -(sum($2:_idxExpr(indices, a)))")
				.setUnidirectional(true)
				.parseGlobalVars("INT:a")
				.parseGlobalVars("INT...:indices")
				.withParsedStatement("sum($1:_idxExpr(indices, -(a)))", hooks)
				.toParsedStatement("-(sum($2:_idxExpr(indices, a)))", hooks)
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx, "sum(_idxExpr(indices, +(ops))) => +(argList(sum(_idxExpr(indices, op1)), sum(_idxExpr(...)), ...))")
				.setUnidirectional(true)
				.parseGlobalVars("INT...:indices")
				.parseGlobalVars("FLOAT...:ops")
				.withParsedStatement("sum($1:_idxExpr(indices, +(ops)))", hooks)
				.toParsedStatement("$4:+($3:argList(sum($2:_idxExpr(indices, +(ops)))))", hooks) // The inner +(ops) is temporary and will be removed
				.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
				.apply(hooks.get(3).getId(), newArgList -> {
					RewriterStatement oldArgList = newArgList.getChild(0, 0, 1, 0);
					newArgList.getChild(0, 0).getOperands().set(1, oldArgList.getChild(0));

					for (int i = 1; i < oldArgList.getOperands().size(); i++) {
						RewriterStatement newIdxExpr = newArgList.getChild(0, 0).copyNode();
						newIdxExpr.getOperands().set(1, oldArgList.getChild(i));
						RewriterStatement newSum = new RewriterInstruction()
								.as(UUID.randomUUID().toString())
								.withInstruction("sum")
								.withOps(newIdxExpr);
						RewriterUtils.copyIndexList(newIdxExpr);
						newIdxExpr.refreshReturnType(ctx);
						newSum.consolidate(ctx);
						newArgList.getOperands().add(newSum);
					}

					newArgList.refreshReturnType(ctx);
				}, true)
				.apply(hooks.get(4).getId(), stmt -> {
					stmt.refreshReturnType(ctx);
				}, true)
				.build()
		);
	}

	public static void flattenedAlgebraRewrites(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		// Minus pushdown
		rules.add(new RewriterRuleBuilder(ctx, "-(+(...)) => +(-(el1), -(el2), ...)")
				.setUnidirectional(true)
				.parseGlobalVars("FLOAT...:ops")
				.withParsedStatement("-(+(ops))", hooks)
				.toParsedStatement("$1:+(ops)", hooks) // Temporary
				.apply(hooks.get(1).getId(), (stmt, match) -> {
					RewriterStatement argList = stmt.getChild(0);

					for (int i = 0; i < argList.getOperands().size(); i++) {
						RewriterInstruction newStmt = new RewriterInstruction().as(UUID.randomUUID().toString()).withInstruction("-").withOps(argList.getOperands().get(i));
						newStmt.consolidate(ctx);
						argList.getOperands().set(i, newStmt);
					}

					// TODO: This is inefficient
					RewriterUtils.tryFlattenNestedOperatorPatterns(ctx, match.getNewExprRoot());
				}, true)
				.build()
		);

		// TODO: Distributive law
	}

	public static void buildElementWiseAlgebraicCanonicalization(final List<RewriterRule> rules, final RuleContext ctx) {
		RewriterUtils.buildTernaryPermutations(List.of("FLOAT", "INT", "BOOL"), (t1, t2, t3) -> {
			rules.add(new RewriterRuleBuilder(ctx, "*(+(a, b), c) => +(*(a, c), *(b, c))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars(t3 + ":c")
					.withParsedStatement("*(+(a, b), c)")
					.toParsedStatement("+(*(a, c), *(b, c))")
					.build()
			);

			rules.add(new RewriterRuleBuilder(ctx, "*(c, +(a, b)) => +(*(c, a), *(c, b))")
					.setUnidirectional(true)
					.parseGlobalVars(t1 + ":a")
					.parseGlobalVars(t2 + ":b")
					.parseGlobalVars(t3 + ":c")
					.withParsedStatement("*(c, +(a, b))")
					.toParsedStatement("+(*(c, a), *(c, b))")
					.build()
			);
		});
	}

	@Deprecated
	public static void streamifyExpressions(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		ALL_TYPES.forEach(t -> {
			if (t.equals("MATRIX"))
				return;

			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":b")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v")
					.withParsedStatement("$1:ElementWiseInstruction($3:_m(i, j, v), b)", hooks)
					.toParsedStatement("$4:_m(i, j, $2:ElementWiseInstruction(v, b))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(3).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.build());

			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A")
					.parseGlobalVars(t + ":b")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v")
					.withParsedStatement("$1:ElementWiseInstruction(b, $3:_m(i, j, v))", hooks)
					.toParsedStatement("$4:_m(i, j, $2:ElementWiseInstruction(b, v))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.link(hooks.get(3).getId(), hooks.get(4).getId(), RewriterStatement::transferMeta)
					.build());
		});


	}

	public static void flattenOperations(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		RewriterUtils.buildBinaryPermutations(List.of("INT", "INT..."), (t1, t2) -> {
			for (String t3 : List.of("FLOAT", "FLOAT*", "INT", "INT*", "BOOL", "BOOL*")) {
				rules.add(new RewriterRuleBuilder(ctx, "Flatten nested index expression")
						.setUnidirectional(true)
						.parseGlobalVars(t1 + ":i")
						.parseGlobalVars(t2 + ":j")
						.parseGlobalVars(t3 + ":v")
						.withParsedStatement("$1:_idxExpr(i, $2:_idxExpr(j, v))", hooks)
						.toParsedStatement("$3:_idxExpr(argList(i, j), v)", hooks)
						.link(hooks.get(1).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
						.apply(hooks.get(3).getId(), (stmt, match) -> {
							UUID newOwnerId = (UUID) stmt.getMeta("ownerId");

							if (newOwnerId == null)
								throw new IllegalArgumentException();

							stmt.getOperands().get(0).getOperands().get(1).unsafePutMeta("ownerId", newOwnerId);
						}, true)
						.build());

				if (t1.equals("INT")) {
					// This must be executed after the rule above
					rules.add(new RewriterRuleBuilder(ctx, "Flatten nested index expression")
							.setUnidirectional(true)
							.parseGlobalVars(t1 + ":i")
							.parseGlobalVars(t3 + ":v")
							.withParsedStatement("$1:_idxExpr(i, v)", hooks)
							.toParsedStatement("$3:_idxExpr(argList(i), v)", hooks)
							.link(hooks.get(1).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
							.build());
				}
			}
		});

		RewriterUtils.buildBinaryPermutations(List.of("MATRIX", "INT", "FLOAT", "BOOL"), (t1, t2) -> {
			//if (RewriterUtils.convertibleType(t1, t2) != null) {
				rules.add(new RewriterRuleBuilder(ctx, "Flatten fusable binary operator")
						.setUnidirectional(true)
						.parseGlobalVars(t1 + ":A")
						.parseGlobalVars(t2 + ":B")
						.withParsedStatement("$1:FusableBinaryOperator(A,B)", hooks)
						.toParsedStatement("$2:FusedOperator(argList(A,B))", hooks)
						.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
						.build());

				rules.add(new RewriterRuleBuilder(ctx, "Flatten fusable binary operator")
						.setUnidirectional(true)
						.parseGlobalVars(t1 + "...:A")
						.parseGlobalVars(t2 + ":B")
						.withParsedStatement("$1:FusableBinaryOperator($2:FusedOperator(A), B)", hooks)
						.toParsedStatement("$3:FusedOperator(argList(A, B))", hooks)
						.iff(match -> {
							return match.getMatchRoot().trueInstruction().equals(match.getMatchRoot().getOperands().get(0).trueInstruction());
						}, true)
						.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
						.build());

				rules.add(new RewriterRuleBuilder(ctx, "Flatten fusable binary operator")
						.setUnidirectional(true)
						.parseGlobalVars(t1 + "...:A")
						.parseGlobalVars(t2 + ":B")
						.withParsedStatement("$1:FusableBinaryOperator(B, $2:FusedOperator(A))", hooks)
						.toParsedStatement("$3:FusedOperator(argList(B, A))", hooks)
						.iff(match -> {
							return match.getMatchRoot().trueInstruction().equals(match.getMatchRoot().getOperands().get(0).trueInstruction());
						}, true)
						.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
						.build());

				/*List.of(t1, t1 + "...").forEach(t -> {
					ALL_TYPES.forEach(mT -> {
						rules.add(new RewriterRuleBuilder(ctx)
								.setUnidirectional(true)
								.parseGlobalVars(t2 + ":A")
								.parseGlobalVars(mT + ":B")
								.parseGlobalVars(t + ":C")
								.withParsedStatement("$1:FusedOperator(argList($2:FusableBinaryOperator(A, B), C))", hooks)
								.toParsedStatement("$3:FusedOperator(argList(argList(A, B), C))", hooks)
								.iff(match -> {
									return match.getMatchRoot().trueInstruction().equals(match.getMatchRoot().getOperands().get(0).getOperands().get(0).trueInstruction());
								}, true)
								.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
								.build());

						rules.add(new RewriterRuleBuilder(ctx)
								.setUnidirectional(true)
								.parseGlobalVars(t2 + ":A")
								.parseGlobalVars(mT + ":B")
								.parseGlobalVars(t + ":C")
								.withParsedStatement("$1:FusedOperator(argList(C, $2:FusableBinaryOperator(A, B)))", hooks)
								.toParsedStatement("$3:FusedOperator(argList(C, argList(A, B)))", hooks)
								.iff(match -> {
									return match.getMatchRoot().trueInstruction().equals(match.getMatchRoot().getOperands().get(0).getOperands().get(1).trueInstruction());
								}, true)
								.link(hooks.get(2).getId(), hooks.get(3).getId(), RewriterStatement::transferMeta)
								.build());

						//System.out.println("Rule: " + rules.get(rules.size()-2));
					});

				});*/
			//}
		});

	}

	@Deprecated
	public static void collapseStreamingExpressions(final List<RewriterRule> rules, final RuleContext ctx) {

		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("sum(_m(_idx(1, nrow(A)), 1, sum(_m(_idx(1, ncol(A)), 1, [](A, i, j)))))", hooks)
				.toParsedStatement("sum(A)", hooks)
				.build()
		);

		RewriterUtils.buildBinaryPermutations(List.of("INT", "FLOAT", "BOOL"), (t1, t2) -> {
			rules.add(new RewriterRuleBuilder(ctx)
					.setUnidirectional(true)
					.parseGlobalVars("MATRIX:A,B")
					.parseGlobalVars("INT:i,j")
					.parseGlobalVars("FLOAT:v1,v2")
					.withParsedStatement("$3:_m(i, j, $1:ElementWiseInstruction(v1, v2))", hooks)
					.toParsedStatement("$2:ElementWiseInstruction($4:_m(i, j, v1), $5:_m(i, j, v2))", hooks)
					.link(hooks.get(1).getId(), hooks.get(2).getId(), RewriterStatement::transferMeta)
					.linkManyUnidirectional(hooks.get(3).getId(), List.of(hooks.get(4).getId(), hooks.get(5).getId()), link -> {
						RewriterStatement.transferMeta(link);

						// Now detach the reference for the second matrix stream

						UUID newId = UUID.randomUUID();
						link.newStmt.get(1).unsafePutMeta("ownerId", newId);
						RewriterStatement idxI = link.newStmt.get(1).getOperands().get(0).copyNode();
						RewriterStatement idxJ = link.newStmt.get(1).getOperands().get(1).copyNode();
						UUID oldIId = (UUID)idxI.getMeta("idxId");
						UUID oldJId = (UUID)idxJ.getMeta("idxId");
						idxI.unsafePutMeta("idxId", UUID.randomUUID());
						idxI.unsafePutMeta("ownerId", newId);
						idxJ.unsafePutMeta("idxId", UUID.randomUUID());
						idxJ.unsafePutMeta("ownerId", newId);

						RewriterUtils.replaceReferenceAware(link.newStmt.get(1), stmt -> {
							UUID idxId = (UUID) stmt.getMeta("idxId");
							if (idxId != null) {
								if (idxId.equals(oldIId))
									return idxI;
								else if (idxId.equals(oldJId))
									return idxJ;
							}

							return null;
						});
					}, true)
					.build()
			);
		});



		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:a,b,c,d")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("_m($1:_idx(a, b), $2:_idx(c, d), [](A, $1, $2))", hooks)
				.toParsedStatement("A", hooks)
				.iff(match -> {
					RewriterStatement A = match.getMatchRoot().getOperands().get(2).getOperands().get(0);
					RewriterStatement a = match.getMatchRoot().getOperands().get(0).getOperands().get(0);
					RewriterStatement b = match.getMatchRoot().getOperands().get(0).getOperands().get(1);
					RewriterStatement c = match.getMatchRoot().getOperands().get(1).getOperands().get(0);
					RewriterStatement d = match.getMatchRoot().getOperands().get(1).getOperands().get(1);

					if (a.isLiteral() && ((long)a.getLiteral()) == 1
						&& b == A.getMeta("nrow")
						&& c.isLiteral() && ((long)c.getLiteral()) == 1
						&& d == A.getMeta("ncol")) {
						return true;
					}

					return false;
				}, true)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:a,b,c,d")
				.parseGlobalVars("FLOAT:v")
				.parseGlobalVars("LITERAL_INT:1")
				.withParsedStatement("_m($1:_idx(a, b), $2:_idx(c, d), [](A, $1, $2))", hooks)
				.toParsedStatement("$3:[](A, a, b, c, d)", hooks)
				.build()
		);

		/*rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("LITERAL_INT:1")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_m(i, j, sum($1:ElementWiseInstruction(A, B)))", hooks)
				.toParsedStatement("sum(A)", hooks)
				.build()
		);*/



		// TODO: The rule below only hold true for i = _idx(1, nrow(i)) and j = _idx(1, ncol(i))
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_m(i, j, [](A, j, i))", hooks)
				.toParsedStatement("t(A)", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_m(i, j, [](A, i, i))", hooks)
				.toParsedStatement("diag(A)", hooks)
				.build()
		);

		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_m(i, j, [](A, j, j))", hooks)
				.toParsedStatement("diag(A)", hooks)
				.build()
		);
	}

	public static void assertCollapsed(final List<RewriterRule> rules, final RuleContext ctx) {
		HashMap<Integer, RewriterStatement> hooks = new HashMap<>();
		rules.add(new RewriterRuleBuilder(ctx)
				.setUnidirectional(true)
				.parseGlobalVars("MATRIX:A,B")
				.parseGlobalVars("INT:i,j")
				.parseGlobalVars("FLOAT:v")
				.withParsedStatement("_m(i, j, v)", hooks)
				.toParsedStatement("$1:_m(i, j, v)", hooks)
				.iff(match -> {
					throw new IllegalArgumentException("Could not eliminate stream expression: " + match.getMatchRoot().toString(ctx));
				}, true)
				.build()
		);
	}

}
