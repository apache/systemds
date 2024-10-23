package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.mutable.MutableObject;
import scala.Tuple3;

import java.util.HashMap;
import java.util.Optional;
import java.util.Set;
import java.util.UUID;
import java.util.function.Function;

public class MetaPropagator implements Function<RewriterStatement, RewriterStatement> {
	private final RuleContext ctx;

	public MetaPropagator(RuleContext ctx) {
		this.ctx = ctx;
	}

	// TODO: Maybe automatically recompute hash codes?
	public RewriterStatement apply(RewriterStatement root) {
		RewriterAssertions assertions = root.getAssertions(ctx);
		MutableObject<RewriterStatement> out = new MutableObject<>(root);
		HashMap<Object, RewriterStatement> literalMap = new HashMap<>();
		root.forEachPostOrder((el, parent, pIdx) -> {
			RewriterStatement toSet = propagateDims(el, parent, pIdx, assertions);

			if (toSet != null) {
				el = toSet;
				if (parent == null)
					out.setValue(toSet);
				else
					parent.getOperands().set(pIdx, toSet);
			}

			// Assert
			if (el.getResultingDataType(ctx).startsWith("MATRIX")
				&& (el.getNCol() == null || el.getNRow() == null))
				throw new IllegalArgumentException("Some properties have not been set by the meta propagator: " + el.toString(ctx) + " :: " + el.getResultingDataType(ctx));


			// Eliminate common literals
			if (el.isLiteral()) {
				RewriterStatement existingLiteral = literalMap.get(el.getLiteral());

				if (existingLiteral != null) {
					if (parent == null)
						out.setValue(existingLiteral);
					else
						parent.getOperands().set(pIdx, existingLiteral);
				} else {
					literalMap.put(el.getLiteral(), el);
				}
			}

			validate(el);
		});

		return out.getValue();
	}

	private RewriterStatement propagateDims(RewriterStatement root, RewriterStatement parent, int pIdx, RewriterAssertions assertions) {
		if (!root.getResultingDataType(ctx).startsWith("MATRIX")) {
			if (root.isInstruction()) {
				String ti = root.trueTypedInstruction(ctx);
				RewriterStatement ret = null;

				switch (ti) {
					case "ncol(MATRIX)":
						ret = (RewriterStatement)root.getOperands().get(0).getMeta("ncol");
						break;
					case "nrow(MATRIX)":
						ret = (RewriterStatement)root.getOperands().get(0).getMeta("nrow");
						break;
				}

				if (ret == null)
					return null;

				return ret;

				/*RewriterStatement asserted = assertions != null ? assertions.getAssertionStatement(ret) : null;

				if (asserted == null || asserted == parent)
					return ret;

				return asserted;*/
			}
			return null;
		}

		Object colAccess;
		Object rowAccess;

		if (root.getOperands() == null || root.getOperands().isEmpty()) {
			if (root.getMeta("ncol") == null)
				root.unsafePutMeta("ncol", new RewriterInstruction().withInstruction("ncol").withOps(root).as(UUID.randomUUID().toString()).consolidate(ctx));
			if (root.getMeta("nrow") == null)
				root.unsafePutMeta("nrow", new RewriterInstruction().withInstruction("nrow").withOps(root).as(UUID.randomUUID().toString()).consolidate(ctx));
			return null;
		}

		if (root.isInstruction()) {
			Optional<RewriterStatement> firstMatrixStatement = root.getOperands().stream().filter(el -> el.getResultingDataType(ctx).startsWith("MATRIX")).findFirst();
			switch(root.trueInstruction()) {
				// Handle generators
				case "rand":
					root.unsafePutMeta("nrow", root.getOperands().get(0));
					root.unsafePutMeta("ncol", root.getOperands().get(1));
					return null;
				case "as.matrix":
					root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L));
					root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L));
					return null;
				case "argList":
					// TODO: We assume argLists always occur if the matrix properties don't change (for now)
					root.unsafePutMeta("nrow", firstMatrixStatement.get().getMeta("nrow"));
					root.unsafePutMeta("ncol", firstMatrixStatement.get().getMeta("ncol"));
					return null;
				case "_map":
					root.unsafePutMeta("nrow", root.getOperands().get(1).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(1).getMeta("ncol"));
					return null;
				case "+":
				case "-":
				case "*":
				case "inv":
				case "==":
				case "!=":
				case "&":
				case "|":
				case "<":
				case ">":
					if (firstMatrixStatement.isEmpty())
						throw new IllegalArgumentException(root.toString(ctx) + " has empty args!");
					root.unsafePutMeta("nrow", firstMatrixStatement.get().getMeta("nrow"));
					root.unsafePutMeta("ncol", firstMatrixStatement.get().getMeta("ncol"));
					return null;
			}

			switch(root.trueTypedInstruction(ctx)) {
				case "t(MATRIX)":
					colAccess = root.getOperands().get(0).getMeta("ncol");
					rowAccess = root.getOperands().get(0).getMeta("nrow");
					root.unsafePutMeta("ncol", rowAccess);
					root.unsafePutMeta("nrow", colAccess);
					return null;
				case "_m(INT,INT,FLOAT)":
					if (root.getOperands().get(0).isInstruction()
							&& root.getOperands().get(0).trueTypedInstruction(ctx).equals("_idx(INT,INT)")) {
						root.unsafePutMeta("nrow", root.getOperands().get(0).getOperands().get(1));
					} else {
						root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					}

					if (root.getOperands().get(1).isInstruction()
							&& root.getOperands().get(1).trueTypedInstruction(ctx).equals("_idx(INT,INT)")) {
						root.unsafePutMeta("ncol", root.getOperands().get(1).getOperands().get(1));
					} else {
						root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L).consolidate(ctx));
					}
					return null;
				case "%*%(MATRIX,MATRIX)":
					rowAccess = root.getOperands().get(0).getMeta("nrow");
					colAccess = root.getOperands().get(1).getMeta("ncol");
					root.unsafePutMeta("nrow", rowAccess);
					root.unsafePutMeta("ncol", colAccess);
					return null;
				case "diag(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L));
					return null;
				case "[](MATRIX,INT,INT,INT,INT)":
					Integer[] ints = new Integer[4];

					for (int i = 0; i < 4; i++)
						if (root.getOperands().get(1).isLiteral())
							ints[i] = (Integer)root.getOperands().get(1).getLiteral();

					if (ints[0] != null && ints[1] != null) {
						root.unsafePutMeta("nrow", ints[1] - ints[0] + 1);
					} else {
						throw new NotImplementedException();
						// TODO:
					}

					if (ints[2] != null && ints[3] != null) {
						root.unsafePutMeta("ncol", ints[3] - ints[2] + 1);
					} else {
						throw new NotImplementedException();
					}

					return null;
				case "rowSums(MATRIX)":
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", new RewriterDataType().ofType("INT").as("1").asLiteral(1L));
					return null;
				case "colSums(MATRIX)":
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
					root.unsafePutMeta("nrow", new RewriterDataType().ofType("INT").as("1").asLiteral(1L));
					return null;
			}

			RewriterInstruction instr = (RewriterInstruction) root;

			if (instr.getProperties(ctx).contains("ElementWiseInstruction")) {
				if (root.getOperands().get(0).getResultingDataType(ctx).startsWith("MATRIX")) {
					root.unsafePutMeta("nrow", root.getOperands().get(0).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(0).getMeta("ncol"));
				} else {
					root.unsafePutMeta("nrow", root.getOperands().get(1).getMeta("nrow"));
					root.unsafePutMeta("ncol", root.getOperands().get(1).getMeta("ncol"));
				}

				return null;
			}

			throw new NotImplementedException("Unknown instruction: " + instr.trueTypedInstruction(ctx) + "\n" + instr.toString(ctx));
		}

		return null;
	}

	private void validate(RewriterStatement stmt) {
		if (stmt.isInstruction()) {
			if (stmt.trueInstruction().equals("_idx") && (stmt.getMeta("ownerId") == null || stmt.getMeta("idxId") == null))
				throw new IllegalArgumentException(stmt.toString(ctx));

			if (stmt.trueInstruction().equals("_m") && stmt.getMeta("ownerId") == null)
				throw new IllegalArgumentException(stmt.toString(ctx));
		}
	}
}
