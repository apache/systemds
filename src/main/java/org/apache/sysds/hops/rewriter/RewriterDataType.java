package org.apache.sysds.hops.rewriter;

import org.apache.commons.collections4.bidimap.DualHashBidiMap;
import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.function.TriFunction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

public class RewriterDataType extends RewriterStatement {
	private String id;
	private String type;
	private Object literal = null;
	private boolean consolidated = false;
	private int hashCode;

	@Override
	public String getId() {
		return id;
	}

	@Override
	public String getResultingDataType(final RuleContext ctx) {
		return type;
	}

	@Override
	public void refreshReturnType(final RuleContext ctx) {}

	@Override
	public boolean isLiteral() {
		return literal != null && !(literal instanceof List<?>);
	}

	@Override
	public Object getLiteral() {
		return literal;
	}

	@Override
	public long intLiteral() {
		if (getLiteral() instanceof Boolean)
			return (boolean)getLiteral() ? 1 : 0;
		return (long)getLiteral();
	}

	@Override
	public double floatLiteral() {
		if (getLiteral() instanceof Boolean)
			return (boolean)getLiteral() ? 1 : 0;
		if (getLiteral() instanceof Long)
			return Double.valueOf((Long)getLiteral());
		return (double)getLiteral();
	}

	@Override
	public boolean boolLiteral() {
		if (getLiteral() instanceof Boolean)
			return (boolean)getLiteral();
		if (getLiteral() instanceof Long)
			return (long)getLiteral() == 0L;
		return (double)getLiteral() == 0.0;
	}

	@Override
	public void setLiteral(Object literal) {
		if (consolidated)
			throw new IllegalArgumentException();

		this.literal = literal;
	}

	@Override
	public RewriterStatement getLiteralStatement() {
		return this;
	}

	@Override
	public boolean isArgumentList() {
		return false;
	}

	@Override
	public List<RewriterStatement> getArgumentList() {
		return null;
	}

	@Override
	public boolean isInstruction() {
		return false;
	}

	@Override
	public boolean isEClass() {
		return false;
	}

	@Override
	public String trueInstruction() {
		return null;
	}

	@Override
	public String trueTypedInstruction(RuleContext ctx) {
		return null;
	}

	@Override
	public RewriterStatement consolidate(final RuleContext ctx) {
		if (consolidated)
			return this;

		if (id == null || id.isEmpty())
			throw new IllegalArgumentException("The id of a data type cannot be empty");
		if (type == null ||type.isEmpty())
			throw new IllegalArgumentException("The type of a data type cannot be empty");

		if (isLiteral())
			hashCode = Objects.hash(-1, -1, type, literal);
		else
			hashCode = Objects.hash(rid, refCtr, type);
		return this;
	}

	@Override
	public int recomputeHashCodes(boolean recursively, final RuleContext ctx) {
		if (isLiteral())
			hashCode = Objects.hash(-1, -1, type, literal);
		else
			hashCode = Objects.hash(rid, refCtr, type);
		return hashCode;
	}

	@Override
	public int structuralHashCode() {
		return hashCode;
	}

	@Override
	public RewriterStatement rename(String id) {
		this.id = id;
		return this;
	}

	@Override
	public int hashCode() {
		if (isLiteral())
			return hashCode;

		return super.hashCode();
	}

	@Override
	public boolean equals(Object o) {
		if (isLiteral())
			return o instanceof RewriterDataType && getLiteral().equals(((RewriterDataType)o).getLiteral());
		return super.equals(o);
	}

	@Override
	public int computeIds(int id) {
		if (!isLiteral())
			return super.computeIds(id);

		rid = -1;
		return id;
	}

	@Override
	public void computeRefCtrs() {
		refCtr = -1;
	}

	@Override
	public boolean isConsolidated() {
		return consolidated;
	}

	@Override
	public boolean match(final MatcherContext mCtx) {
		RewriterStatement stmt = mCtx.currentStatement;
		RuleContext ctx = mCtx.ctx;
		String dType = stmt.getResultingDataType(ctx);

		if (!(stmt instanceof RewriterDataType) && !mCtx.statementsCanBeVariables) {
			mCtx.setFirstMismatch(this, stmt);
			return false;
		}

		if (!dType.equals(type)) {
			if (!mCtx.allowTypeHierarchy) {
				mCtx.setFirstMismatch(this, stmt);
				return false;
			}

			Set<String> types = ctx.typeHierarchy.get(dType);
			if (types == null || !types.contains(type)) {
				mCtx.setFirstMismatch(this, stmt);
				return false;
			}
		}

		// TODO: This way of literal matching might cause confusion later on
		if (mCtx.literalsCanBeVariables) {
			if (isLiteral())
				if (!mCtx.ignoreLiteralValues && (!stmt.isLiteral() || !getLiteral().equals(stmt.getLiteral()))) {
					mCtx.setFirstMismatch(this, stmt);
					return false;
				}
		} else {
			if (isLiteral() != stmt.isLiteral()) {
				mCtx.setFirstMismatch(this, stmt);
				return false;
			}
			if (!mCtx.ignoreLiteralValues && isLiteral() && !getLiteral().equals(stmt.getLiteral())) {
				mCtx.setFirstMismatch(this, stmt);
				return false;
			}
		}

		RewriterStatement assoc = mCtx.getDependencyMap().get(this);
		if (assoc == null) {
			if (!mCtx.allowDuplicatePointers && mCtx.getDependencyMap().containsValue(stmt)) {
				mCtx.setFirstMismatch(this, stmt);
				return false; // Then the statement variable is already associated with another variable
			}
			mCtx.getDependencyMap().put(this, stmt);
			return true;
		} else if (assoc.equals(stmt)) {
			return true;
		}

		mCtx.setFirstMismatch(this, stmt);
		return false;
	}

	@Override
	public RewriterStatement clone() {
		return new RewriterDataType().as(id).ofType(type);
	}

	@Override
	public RewriterStatement copyNode() {
		return new RewriterDataType().as(id).ofType(type).asLiteral(literal);
	}

	@Override
	public RewriterStatement nestedCopyOrInject(Map<RewriterStatement, RewriterStatement> copiedObjects, TriFunction<RewriterStatement, RewriterStatement, Integer, RewriterStatement> injector, RewriterStatement parent, int pIdx) {
		RewriterStatement mCpy = copiedObjects.get(this);
		if (mCpy != null)
			return mCpy;
		mCpy = injector.apply(this, parent, pIdx);
		if (mCpy != null) {
			// Then change the reference to the injected object
			copiedObjects.put(this, mCpy);
			return mCpy;
		}

		RewriterDataType mCopy = new RewriterDataType();
		mCopy.id = id;
		mCopy.type = type;
		if (literal != null && literal instanceof List<?>) {
			final ArrayList<Object> mList = new ArrayList<>(((List<?>)literal).size());
			mCopy.literal = mList;
			((List<?>) literal).forEach(el -> {
				if (el instanceof RewriterStatement)
					mList.add(((RewriterStatement)el).nestedCopyOrInject(copiedObjects, injector));
			});
		} else
			mCopy.literal = literal;
		mCopy.consolidated = consolidated;
		mCopy.hashCode = hashCode;
		if (meta != null)
			mCopy.meta = new HashMap<>(meta);
		copiedObjects.put(this, mCopy);
		mCopy.nestedCopyOrInjectMetaStatements(copiedObjects, injector);

		return mCopy;
	}

	@Override
	public RewriterStatement simplify(final RuleContext ctx) {
		return this;
	}

	public String getType() {
		return type;
	}

	@Override
	public RewriterDataType as(String id) {
		if (consolidated)
			throw new IllegalArgumentException("A data type cannot be modified after consolidation");
		this.id = id;
		return this;
	}

	public RewriterDataType ofType(String type) {
		if (consolidated)
			throw new IllegalArgumentException("A data type cannot be modified after consolidation");
		this.type = type;
		return this;
	}

	public RewriterDataType asLiteral(Object literal) {
		if (consolidated)
			throw new IllegalArgumentException("A data type cannot be modified after consolidation");
		this.literal = literal;
		return this;
	}

	@Override
	public int toParsableString(StringBuilder sb, Map<RewriterStatement, Integer> refs, int maxRefId, Map<String, Set<String>> vars, final RuleContext ctx) {
		String mType = type;
		String varStr = id;

		if (isLiteral()) {
			mType = "LITERAL_" + type;
			varStr = getLiteral().toString();

			if (getLiteral() instanceof Boolean)
				varStr = varStr.toUpperCase();
		}

		Set<String> varSet = vars.get(mType);

		if (varSet == null) {
			varSet = new HashSet<>();
			vars.put(mType, varSet);
		}

		varSet.add(varStr);
		sb.append(varStr);

		return maxRefId;
	}

	@Override
	public String toString(final RuleContext ctx) {
		if (!isLiteral())
			return getId() + "::" + getResultingDataType(ctx) + "[" + hashCode() + "]";

		if (getLiteral() instanceof Boolean)
			return getLiteral().toString().toUpperCase();

		return getLiteral().toString() + "::" + getResultingDataType(ctx) + "["  + hashCode() + "]";
	}
}
