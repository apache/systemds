package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.function.TriFunction;
import org.apache.commons.lang3.mutable.MutableObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class RewriterDataTypeSet extends RewriterStatement {

	private RewriterDataType result = new RewriterDataType();
	private HashSet<RewriterStatement> operands = new HashSet<>();

	@Override
	public String getId() {
		return result.getId();
	}

	@Override
	public String getResultingDataType(RuleContext ctx) {
		return "SET[" + result.getResultingDataType(ctx) + "]";
	}

	@Override
	public boolean isLiteral() {
		return operands.stream().allMatch(RewriterStatement::isLiteral);
	}

	@Override
	public Object getLiteral() {
		return operands;
	}

	@Override
	public RewriterStatement consolidate(RuleContext ctx) {
		// This object does not support consolidation as it is a legacy concept
		// It will forward it to the other objects though
		operands.forEach(stmt -> stmt.consolidate(ctx));
		return this;
	}

	@Override
	public boolean isConsolidated() {
		return result.isConsolidated() && operands.stream().allMatch(RewriterStatement::isConsolidated);
	}

	@Override
	public RewriterStatement clone() {
		// Not supported for now
		throw new NotImplementedException();
	}

	@Override
	public RewriterStatement copyNode() {
		RewriterDataTypeSet cpy = new RewriterDataTypeSet();
		if (meta != null)
			cpy.meta = new HashMap<>(meta);
		cpy.result = (RewriterDataType)result.copyNode();
		cpy.operands = new HashSet<>();
		return null;
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

		RewriterDataTypeSet mCopy = new RewriterDataTypeSet();
		mCopy.result = (RewriterDataType)result.copyNode();
		mCopy.operands = new HashSet<>(operands.size());
		if (meta != null)
			mCopy.meta = new HashMap<>(meta);
		else
			mCopy.meta = null;
		copiedObjects.put(this, mCopy);

		operands.forEach(el -> mCopy.operands.add(el.nestedCopyOrInject(copiedObjects, injector, mCopy, -1)));

		return mCopy;
	}

	// This will return the first 'random' match if there are multiple options
	// E.g. if we match (a + b * c) with a statement (a * b + b * c) it will either return the association ((a * b) + b * c) or (a * b + (b * c))
	@Override
	public boolean match(MatcherContext matcherContext) {
		throw new IllegalArgumentException();
		/*final RuleContext ctx = matcherContext.ctx;
		final RewriterStatement stmt = matcherContext.currentStatement;
		if (stmt instanceof RewriterDataTypeSet && getResultingDataType(ctx).equals(stmt.getResultingDataType(ctx))) {
			RewriterDataTypeSet dts = (RewriterDataTypeSet) stmt;

			// Check if we can use the hash signature of the object
			if (!matcherContext.allowDuplicatePointers && !matcherContext.allowPropertyScan && !matcherContext.allowTypeHierarchy) {
				// Then we can use the hash signature to speed up the process
				// TODO: This is for later
				System.out.println("[WARN] Matching is currently slow for sets!");
			}

			if (this.operands.size() != dts.operands.size())
				return false;

			RewriterStatement existingRef = matcherContext.findInternalReference(new RewriterRule.IdentityRewriterStatement(this));
			if (existingRef != null)
				return existingRef == stmt;

			RewriterRule.LinkObject ruleLink = matcherContext.ruleLinks.get(this);

			if (ruleLink != null)
				matcherContext.getLinks().add(new RewriterRule.ExplicitLink(dts, ruleLink.stmt, ruleLink.transferFunction));

			// Check all possible permutations, but allow early stopping
			MutableObject<MatcherContext> checkpoint = new MutableObject<>(matcherContext.createCheckpoint());
			boolean symmetric = !matcherContext.allowDuplicatePointers && !matcherContext.allowPropertyScan && !matcherContext.allowTypeHierarchy;

			List<RewriterStatement> ownOperands = new ArrayList<>(operands);
			return RewriterUtils.findMatchingOrderings(ownOperands, new ArrayList<>(dts.operands), new RewriterStatement[dts.operands.size()],
					(thisStmt, toMatch) -> {
						checkpoint.getValue().currentStatement = toMatch;
						boolean matching = thisStmt.match(checkpoint.getValue());

						if (matching) {

						}

						checkpoint.setValue(matcherContext.createCheckpoint());

						return matching;
					},
					matchingPermutation -> {
						matcherContext.getInternalReferences().put(new RewriterRule.IdentityRewriterStatement(this), stmt);
						matcherContext.addSubMatch(checkpoint);
						return matcherContext.terminateOnFirstMatch;
					}, symmetric);
		}

		return false;*/
	}

	@Override
	public int recomputeHashCodes(boolean recursively) {
		if (recursively) {
			// Here we must trigger a re-init of the HashSet, as the hash-values might have changed
			List<RewriterStatement> tempStorage = new ArrayList<>(operands);
			operands.clear();
			operands.addAll(tempStorage);
		}

		return hashCode();
	}

	@Override
	public long getCost() {
		return operands.stream().mapToLong(RewriterStatement::getCost).sum();
	}

	@Override
	public RewriterStatement simplify(RuleContext ctx) {
		// This does nothing
		return this;
	}

	@Override
	public RewriterStatement as(String id) {
		result.as(id);
		return this;
	}

	public RewriterStatement ofContentType(String type) {
		result.ofType(type);
		return this;
	}

	@Override
	public String toString(RuleContext ctx) {
		return "{ " + operands.stream().map(el -> el.toString(ctx)).collect(Collectors.joining(", ")) + "}";
	}

	@Override
	public boolean isArgumentList() {
		// TODO: Is it an argument list?
		return false;
	}

	@Override
	public List<RewriterStatement> getArgumentList() {
		// TODO: Is it an argument list?
		return null;
	}

	@Override
	public boolean isInstruction() {
		// TODO: Is it an instruction?
		return false;
	}

	@Override
	public String trueInstruction() {
		return "_set";
	}

	@Override
	public String trueTypedInstruction(RuleContext ctx) {
		return "_set(" + operands.stream().map(el -> el.getResultingDataType(ctx)).collect(Collectors.joining(",")) + ")";
	}

	@Override
	public int hashCode() {
		return operands.hashCode();
	}

	public void addOp(RewriterStatement stmt, RuleContext ctx) {
		if (!result.getResultingDataType(ctx).equals(stmt.getResultingDataType(ctx)))
			throw new IllegalArgumentException();

		operands.add(stmt);
	}
}
