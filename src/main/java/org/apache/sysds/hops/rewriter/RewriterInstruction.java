package org.apache.sysds.hops.rewriter;

import org.apache.commons.lang3.function.TriFunction;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterInstruction extends RewriterStatement {

	private String id;
	private String returnType;
	private String instr;
	//private RewriterDataType result = new RewriterDataType();
	private ArrayList<RewriterStatement> operands = new ArrayList<>();
	private Function<List<RewriterStatement>, Long> costFunction = null;
	private boolean consolidated = false;
	private int hashCode;

	@Override
	public String getId() {
		return id;
	}

	@Override
	public String getResultingDataType(final RuleContext ctx) {
		if (returnType != null)
			return returnType;

		if (isArgumentList())
			returnType = getOperands().stream().map(op -> op.getResultingDataType(ctx)).reduce(RewriterUtils::defaultTypeHierarchy).get() + "...";
		else
			returnType = ctx.instrTypes.get(trueTypedInstruction(ctx));//getResult(ctx).getResultingDataType(ctx);

		return returnType;
	}

	@Override
	public void refreshReturnType(final RuleContext ctx) {
		returnType = null;
	}

	@Override
	public boolean isLiteral() {
		return false;
	}

	@Override
	public Object getLiteral() {
		return null;
	}

	@Override
	public RewriterStatement consolidate(final RuleContext ctx) {
		if (consolidated)
			return this;

		if (instr == null || instr.isEmpty())
			throw new IllegalArgumentException("Instruction type cannot be empty");

		if (getCostFunction(ctx) == null)
			throw new IllegalArgumentException("Could not find a matching cost function for " + typedInstruction(ctx));

		for (RewriterStatement operand : operands)
			operand.consolidate(ctx);

		//getResult(ctx).consolidate(ctx);

		hashCode = Objects.hash(rid, refCtr, instr, getResultingDataType(ctx), operands);
		consolidated = true;

		return this;
	}
	@Override
	public int recomputeHashCodes(boolean recursively, final RuleContext ctx) {
		if (recursively) {
			//result.recomputeHashCodes(true, ctx);
			operands.forEach(op -> op.recomputeHashCodes(true, ctx));
		}

		hashCode = Objects.hash(rid, refCtr, instr, getResultingDataType(ctx), operands.stream().map(RewriterStatement::structuralHashCode).collect(Collectors.toList()));
		return hashCode;
	}

	@Override
	public boolean isConsolidated() {
		return consolidated;
	}

	@Override
	public boolean match(final MatcherContext mCtx) {
		RewriterStatement stmt = mCtx.currentStatement;
		RuleContext ctx = mCtx.ctx;

		if (mCtx.isDebug())
			System.out.println("Matching: " + this.toString(ctx) + " <=> " + stmt.toString(ctx));

		if (stmt instanceof RewriterInstruction && getResultingDataType(ctx).equals(stmt.getResultingDataType(ctx))) {
			RewriterInstruction inst = (RewriterInstruction)stmt;

			if(!inst.instr.equals(this.instr)) {
				if (!mCtx.allowPropertyScan) {
					mCtx.setFirstMismatch(this, stmt);
					return false;
				}
				Set<String> props = inst.getProperties(ctx);

				if (props == null || !props.contains(typedInstruction(ctx))) {
					mCtx.setFirstMismatch(this, stmt);
					return false;
				}
			}
			if (this.operands.size() != inst.operands.size()) {
				mCtx.setFirstMismatch(this, stmt);
				return false;
			}

			RewriterStatement existingRef = mCtx.findInternalReference(this);

			if (existingRef != null) {
				if (existingRef == stmt)
					return true;
				else {
					mCtx.setFirstMismatch(this, stmt);
					return false;
				}
			}

			if (!mCtx.allowDuplicatePointers && mCtx.getInternalReferences().containsValue(stmt)) {
				mCtx.setFirstMismatch(this, stmt);
				return false;
			}

			RewriterRule.LinkObject ruleLink = mCtx.ruleLinks.get(this);

			if (ruleLink != null)
				mCtx.getLinks().add(new RewriterRule.ExplicitLink(inst, ruleLink.stmt, ruleLink.transferFunction));

			int s = inst.operands.size();

			for (int i = 0; i < s; i++) {
				mCtx.currentStatement = inst.operands.get(i);

				if (!operands.get(i).match(mCtx))
					return false;
			}

			mCtx.getInternalReferences().put(this, stmt);

			return true;
		}

		mCtx.setFirstMismatch(this, stmt);
		return false;
	}

	@Override
	public RewriterStatement copyNode() {
		RewriterInstruction mCopy = new RewriterInstruction();
		mCopy.instr = instr;
		//mCopy.result = (RewriterDataType)result.copyNode();
		mCopy.id = id;
		mCopy.costFunction = costFunction;
		mCopy.consolidated = consolidated;
		mCopy.operands = new ArrayList<>(operands);
		if (meta != null)
			mCopy.meta = new HashMap<>(meta);
		else
			mCopy.meta = null;
		return mCopy;
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

		RewriterInstruction mCopy = new RewriterInstruction();
		mCopy.instr = instr;
		//mCopy.result = (RewriterDataType)result.copyNode();
		mCopy.id = id;
		mCopy.costFunction = costFunction;
		mCopy.consolidated = consolidated;
		mCopy.operands = new ArrayList<>(operands.size());
		mCopy.hashCode = hashCode;
		if (meta != null)
			mCopy.meta = new HashMap<>(meta);
		else
			mCopy.meta = null;
		mCopy.nestedCopyOrInjectMetaStatements(copiedObjects, injector);
		copiedObjects.put(this, mCopy);

		for (int i = 0; i < operands.size(); i++)
			mCopy.operands.add(operands.get(i).nestedCopyOrInject(copiedObjects, injector, mCopy, i));
		//operands.forEach(op -> mCopy.operands.add(op.nestedCopyOrInject(copiedObjects, injector)));

		return mCopy;
	}

	@Override
	public boolean isArgumentList() {
		return trueInstruction().equals("argList");
	}

	@Override
	public List<RewriterStatement> getArgumentList() {
		return isArgumentList() ? getOperands() : null;
	}

	@Override
	public boolean isInstruction() {
		return true;
	}

	@Override
	public RewriterStatement clone() {
		RewriterInstruction mClone = new RewriterInstruction();
		mClone.instr = instr;
		//mClone.result = (RewriterDataType)result.clone();
		mClone.id = id;
		ArrayList<RewriterStatement> clonedOperands = new ArrayList<>(operands.size());

		for (RewriterStatement stmt : operands)
			clonedOperands.add(stmt.clone());

		mClone.operands = clonedOperands;
		mClone.costFunction = costFunction;
		mClone.consolidated = consolidated;
		mClone.meta = meta;
		return mClone;
	}

	/*public void injectData(final RuleContext ctx, RewriterInstruction origData) {
		instr = origData.instr;
		result = (RewriterDataType)origData.getResult(ctx).copyNode();
		operands = new ArrayList<>(origData.operands);
		costFunction = origData.costFunction;
		meta = origData.meta;
	}*/

	/*public RewriterInstruction withLinks(DualHashBidiMap<RewriterStatement, RewriterStatement> links) {
		this.links = links;
		return this;
	}

	public DualHashBidiMap<RewriterStatement, RewriterStatement> getLinks() {
		return links;
	}*/

	@Override
	public List<RewriterStatement> getOperands() {
		return operands == null ? Collections.emptyList() : operands;
	}


	@Override
	public RewriterStatement simplify(final RuleContext ctx) {
		for (int i = 0; i < operands.size(); i++) {
			RewriterStatement stmt = operands.get(i).simplify(ctx);
			if (stmt != null)
				operands.set(i, stmt);
		}

		Function<RewriterInstruction, RewriterStatement> rule = ctx.simplificationRules.get(typedInstruction(ctx));
		if (rule != null) {
			RewriterStatement stmt = rule.apply(this);

			if (stmt != null)
				return stmt;
		}
		return this;
	}

	public RewriterInstruction withInstruction(String instr) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		this.instr = instr;
		return this;
	}

	public RewriterInstruction withOps(RewriterStatement... operands) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		this.operands = new ArrayList<>(Arrays.asList(operands));
		return this;
	}

	public RewriterInstruction addOp(String id) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		this.operands.add(new RewriterDataType().as(id));
		return this;
	}

	public RewriterInstruction addOp(RewriterStatement operand) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		this.operands.add(operand);
		return this;
	}

	public RewriterInstruction ofType(String type) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		RewriterStatement stmt = this.operands.get(this.operands.size()-1);

		if (stmt instanceof RewriterDataType)
			((RewriterDataType)stmt).ofType(type);
		else
			throw new IllegalArgumentException("Can only set the data type of RewriterDataType class");

		return this;
	}

	public Function<List<RewriterStatement>, Long> getCostFunction(final RuleContext ctx) {
		if (this.costFunction == null)
			this.costFunction = ctx.instrCosts.get(typedInstruction(ctx));

		return this.costFunction;
	}

	public RewriterInstruction withCostFunction(Function<List<RewriterStatement>, Long> costFunction) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		this.costFunction = costFunction;
		return this;
	}

	public Optional<RewriterStatement> findOperand(String id) {
		return this.operands.stream().filter(op -> op.getId().equals(id)).findFirst();
	}

	@Override
	public RewriterInstruction as(String id) {
		if (consolidated)
			throw new IllegalArgumentException("An instruction cannot be modified after consolidation");
		this.id = id;
		return this;
	}

	/*public RewriterDataType getResult(final RuleContext ctx) {
		if (this.result.getType() == null) {
			String type = ctx.instrTypes.get(typedInstruction(ctx));

			if (type == null)
				throw new IllegalArgumentException("Type mapping cannot be found for instruction: " + type);

			this.result.ofType(type);
		}

		return this.result;
	}*/

	public String typedInstruction(final RuleContext ctx) {
		return typedInstruction(this.instr, ctx);
	}

	public String getInstr() {
		return instr;
	}

	private String typedInstruction(String instrName, final RuleContext ctx) {
		StringBuilder builder = new StringBuilder();
		builder.append(instrName);
		builder.append("(");

		if (!operands.isEmpty())
			builder.append(operands.get(0).getResultingDataType(ctx));

		if (!isArgumentList()) {
			for (int i = 1; i < operands.size(); i++) {
				builder.append(",");
				builder.append(operands.get(i).getResultingDataType(ctx));
			}
		}

		builder.append(")");
		return builder.toString();
	}

	@Override
	public int toParsableString(StringBuilder sb, Map<RewriterRule.IdentityRewriterStatement, Integer> refs, int maxRefId, Map<String, Set<String>> vars, final RuleContext ctx) {
		RewriterRule.IdentityRewriterStatement id = new RewriterRule.IdentityRewriterStatement(this);
		Integer ref = refs.get(id);

		if (ref != null) {
			sb.append('$');
			sb.append(ref);
			return maxRefId;
		}

		if (refCtr > 1) {
			maxRefId++;
			sb.append('$');
			sb.append(maxRefId);
			sb.append(':');
			refs.put(id, maxRefId);
		}

		sb.append(instr);
		sb.append('(');

		for (int i = 0; i < getOperands().size(); i++) {
			if (i > 0)
				sb.append(',');

			RewriterStatement op = getOperands().get(i);
			maxRefId = op.toParsableString(sb, refs, maxRefId, vars, ctx);
		}

		sb.append(')');

		return maxRefId;
	}

	@Override
	public String toString(final RuleContext ctx) {
		Object varName = getMeta(META_VARNAME);
		if (varName != null)
			return varName.toString();

		Object trueInstrObj = getMeta("trueInstr");
		String typedInstr = trueInstrObj != null ? typedInstruction((String)trueInstrObj, ctx) : typedInstruction(ctx);
		BiFunction<RewriterStatement, RuleContext, String> customStringFunc = ctx.customStringRepr.get(typedInstr);
		if (customStringFunc != null)
			return customStringFunc.apply(this, ctx);

		String instrName = meta == null ? instr : meta.getOrDefault("trueName", instr).toString();

		/*if (operands.size() == 2 && ctx.writeAsBinaryInstruction.contains(instrName))
			return "(" + operands.get(0) + " " + instrName + " " + operands.get(1) + ")";*/

		StringBuilder builder = new StringBuilder();
		builder.append(instrName);
		builder.append("(");
		for (int i = 0; i < operands.size(); i++) {
			if (i > 0)
				builder.append(", ");
			builder.append(operands.get(i).toString(ctx));
		}
		builder.append(")");
		//if (builder.toString().equals("ncol(B::MATRIX)"))
			return builder.toString() + "[" + System.identityHashCode(this) + "]";
		//return builder.toString() + "::" + getResultingDataType(ctx);
	}

	@Override
	public int structuralHashCode() {
		return hashCode;
	}

	@Override
	public long getCost() {
		if (costFunction == null)
			throw new NullPointerException("No cost function has been defined for the instruction: '" + instr + "'");
		long cost = costFunction.apply(operands);
		for (RewriterStatement stmt : operands)
			cost += stmt.getCost();
		return cost;
	}

	public String changeConsolidatedInstruction(String newName, final RuleContext ctx) {
		String typedInstruction = newName;
		String newInstrReturnType = ctx.instrTypes.get(typedInstruction);
		if (newInstrReturnType == null || !newInstrReturnType.equals(getResultingDataType(ctx)))
			throw new IllegalArgumentException("An instruction name can only be changed if it has the same signature (return type) [" + typedInstruction + "::" + newInstrReturnType + " <-> " + typedInstruction(ctx) + "::" + getResultingDataType(ctx) + "]");
		String oldName = instr;
		instr = newName.substring(0, newName.indexOf('('));
		recomputeHashCodes(false, ctx);
		return oldName;
	}

	public boolean hasProperty(String property, final RuleContext ctx) {
		Set<String> properties = getProperties(ctx);

		if (properties == null)
			return false;

		return properties.contains(property);
	}

	public String trueInstruction() {
		// Legacy code
		/*Object trueInstrObj = getMeta("trueInstr");
		if (trueInstrObj != null && trueInstrObj instanceof String)
			return (String)trueInstrObj;*/
		return instr;
	}

	public String trueTypedInstruction(final RuleContext ctx) {
		return typedInstruction(trueInstruction(), ctx);
	}

	public Set<String> getProperties(final RuleContext ctx) {
		Set<String> ret = ctx.instrProperties.get(trueTypedInstruction(ctx));
		if (ret == null)
			return Collections.emptySet();
		return ret;
	}

	void unsafeSetInstructionName(String str) {
		this.instr = str;
	}

}
