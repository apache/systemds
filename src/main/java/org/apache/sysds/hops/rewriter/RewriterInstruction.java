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

import org.apache.commons.lang3.function.TriFunction;
import org.apache.sysds.hops.rewriter.assertions.RewriterAssertions;
import org.apache.sysds.hops.rewriter.rule.RewriterRule;
import org.apache.sysds.hops.rewriter.utils.RewriterUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.UUID;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class RewriterInstruction extends RewriterStatement {

	private String id;
	private String returnType;
	private String instr;
	private ArrayList<RewriterStatement> operands = new ArrayList<>();
	private Function<List<RewriterStatement>, Long> costFunction = null;
	private boolean consolidated = false;
	private int hashCode;

	public RewriterInstruction() {
	}

	public RewriterInstruction(String instr, final RuleContext ctx, RewriterStatement... ops) {
		id = UUID.randomUUID().toString();
		this.instr = instr;
		withOps(ops);
		consolidate(ctx);
	}

	@Override
	protected void compress(RewriterAssertions assertions) {
		id = null;
		operands.trimToSize();
		meta = null;
	}

	@Override
	public String getId() {
		if (isDataOrigin()) {
			if (trueInstruction().equals("const")) {
				boolean regen = id == null;
				if (!regen) {
					try {
						UUID.fromString(id);
						regen = true;
					} catch (Exception e) {
					}
				}
				if (regen) {
					id = "mConst" + new Random().nextInt(10000);
				}
			} else {
				return getChild(0).getId();
			}
		}

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

		if (returnType == null)
			throw new IllegalArgumentException("Return type not found for: " + trueTypedInstruction(ctx));

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
	public RewriterStatement getLiteralStatement() {
		for (RewriterStatement op : getChild(0).getOperands())
			if (op.isLiteral())
				return op;

		return null;
	}

	@Override
	public long intLiteral(boolean cast) {
		throw new UnsupportedOperationException();
	}

	@Override
	public double floatLiteral() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean boolLiteral() {
		throw new UnsupportedOperationException();
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

		hashCode = Objects.hash(rid, refCtr, instr, getResultingDataType(ctx), operands);
		consolidated = true;

		return this;
	}
	@Override
	public int recomputeHashCodes(boolean recursively, final RuleContext ctx) {
		if (recursively) {
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

		// Check for some meta information
		if (mCtx.statementsCanBeVariables && getResultingDataType(ctx).equals("MATRIX")) {
			if ((trueInstruction().equals("rowVec") && stmt.isRowVector())
				|| (trueInstruction().equals("colVec") && stmt.isColVector())) {
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

				mCtx.getInternalReferences().put(this, stmt);

				if (stmt.isInstruction() && (stmt.trueInstruction().equals("rowVec") || stmt.trueInstruction().equals("colVec")))
					mCtx.getDependencyMap().put(getChild(0), stmt.getChild(0));
				else
					mCtx.getDependencyMap().put(getChild(0), stmt);


				return true;
			}
		}

		if (stmt instanceof RewriterInstruction && (getResultingDataType(ctx).equals(stmt.getResultingDataType(ctx)) || (mCtx.allowImplicitTypeConversions && RewriterUtils.isImplicitlyConvertible(stmt.getResultingDataType(ctx), getResultingDataType(ctx))))) {
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

			if (mCtx.findMinimalMismatchRoot) {
				int mismatchCtr = 0;

				for (int i = 0; i < s; i++) {
					mCtx.currentStatement = inst.operands.get(i);

					if (!operands.get(i).match(mCtx))
						mismatchCtr++;
				}

				if (mismatchCtr == 0)
					mCtx.getInternalReferences().put(this, stmt);
				else if (mismatchCtr > 1)
					mCtx.setFirstMismatch(this, stmt);

				return mismatchCtr == 0;
			} else {
				for (int i = 0; i < s; i++) {
					mCtx.currentStatement = inst.operands.get(i);

					if (!operands.get(i).match(mCtx)) {
						if (mCtx.isDebug())
							System.out.println("Mismatch: " + operands.get(i) + " <=> " + inst.operands.get(i));
						return false;
					}
				}

				mCtx.getInternalReferences().put(this, stmt);
				return true;
			}
		}

		mCtx.setFirstMismatch(this, stmt);
		return false;
	}

	@Override
	public RewriterStatement copyNode() {
		RewriterInstruction mCopy = new RewriterInstruction();
		mCopy.instr = instr;
		mCopy.id = id;
		mCopy.costFunction = costFunction;
		mCopy.consolidated = consolidated;
		mCopy.operands = new ArrayList<>(operands);
		mCopy.returnType = returnType;
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
		mCopy.id = id;
		mCopy.costFunction = costFunction;
		mCopy.consolidated = consolidated;
		mCopy.operands = new ArrayList<>(operands.size());
		mCopy.returnType = returnType;
		mCopy.hashCode = hashCode;
		if (meta != null)
			mCopy.meta = new HashMap<>(meta);
		else
			mCopy.meta = null;
		mCopy.nestedCopyOrInjectMetaStatements(copiedObjects, injector);
		copiedObjects.put(this, mCopy);

		for (int i = 0; i < operands.size(); i++)
			mCopy.operands.add(operands.get(i).nestedCopyOrInject(copiedObjects, injector, mCopy, i));

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
	public boolean isEClass() {
		return trueInstruction().equals("_EClass");
	}

	@Deprecated
	@Override
	public RewriterStatement clone() {
		RewriterInstruction mClone = new RewriterInstruction();
		mClone.instr = instr;
		mClone.id = id;
		ArrayList<RewriterStatement> clonedOperands = new ArrayList<>(operands.size());

		for (RewriterStatement stmt : operands)
			clonedOperands.add(stmt.clone());

		mClone.operands = clonedOperands;
		mClone.costFunction = costFunction;
		mClone.consolidated = consolidated;
		mClone.returnType = returnType;
		mClone.meta = meta;
		return mClone;
	}

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

	public String typedInstruction(final RuleContext ctx) {
		return typedInstruction(this.instr, false, ctx);
	}

	public String getInstr() {
		return instr;
	}

	private String typedInstruction(String instrName, boolean allowImplicitConversions, final RuleContext ctx) {
		StringBuilder builder = new StringBuilder();
		builder.append(instrName);
		builder.append("(");

		if (!operands.isEmpty()) {
			String resultingDataType = operands.get(0).getResultingDataType(ctx);
			if (allowImplicitConversions)
				resultingDataType = RewriterUtils.convertImplicitly(resultingDataType);
			builder.append(resultingDataType);
		}

		if (!isArgumentList()) {
			for (int i = 1; i < operands.size(); i++) {
				builder.append(",");
				String resultingDataType = operands.get(i).getResultingDataType(ctx);
				if (allowImplicitConversions)
					resultingDataType = RewriterUtils.convertImplicitly(resultingDataType);
				builder.append(resultingDataType);
			}
		}

		builder.append(")");
		return builder.toString();
	}

	@Override
	public int toParsableString(StringBuilder sb, Map<RewriterStatement, Integer> refs, int maxRefId, Map<String, Set<String>> vars, Set<RewriterStatement> forceCreateRefs, final RuleContext ctx) {
		Integer ref = refs.get(this);

		if (ref != null) {
			sb.append('$');
			sb.append(ref);
			return maxRefId;
		}

		if (refCtr > 1 || forceCreateRefs.contains(this)) {
			maxRefId++;
			sb.append('$');
			sb.append(maxRefId);
			sb.append(':');
			refs.put(this, maxRefId);
		}

		sb.append(instr);
		sb.append('(');

		for (int i = 0; i < getOperands().size(); i++) {
			if (i > 0)
				sb.append(',');

			RewriterStatement op = getOperands().get(i);
			maxRefId = op.toParsableString(sb, refs, maxRefId, vars, forceCreateRefs, ctx);
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
		String typedInstr = trueInstrObj != null ? typedInstruction((String)trueInstrObj, false, ctx) : typedInstruction(ctx);
		BiFunction<RewriterStatement, RuleContext, String> customStringFunc = ctx.customStringRepr.get(typedInstr);
		if (customStringFunc != null)
			return customStringFunc.apply(this, ctx);

		String instrName = meta == null ? instr : meta.getOrDefault("trueName", instr).toString();

		StringBuilder builder = new StringBuilder();
		builder.append(instrName);
		builder.append("(");
		for (int i = 0; i < operands.size(); i++) {
			if (i > 0)
				builder.append(", ");
			builder.append(operands.get(i).toString(ctx));
		}
		builder.append(")");
		return builder + "[" + System.identityHashCode(this) + "]";
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
		return instr;
	}

	public String trueTypedInstruction(final RuleContext ctx) {
		return typedInstruction(trueInstruction(), false, ctx);
	}

	public String trueTypedInstruction(boolean allowImplicitConversions, final RuleContext ctx) {
		return typedInstruction(trueInstruction(), allowImplicitConversions, ctx);
	}

	public Set<String> getProperties(final RuleContext ctx) {
		Set<String> ret = ctx.instrProperties.get(trueTypedInstruction(ctx));
		if (ret == null)
			return Collections.emptySet();
		return ret;
	}

	public void unsafeSetInstructionName(String str) {
		this.instr = str;
	}

}
