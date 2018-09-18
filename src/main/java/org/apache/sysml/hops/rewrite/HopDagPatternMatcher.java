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
package org.apache.sysml.hops.rewrite;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.OpOpDnn;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.utils.Explain;

/**
 * Please see org.apache.sysml.hops.rewrite.RewriteGPUSpecificOps class for usage and design documentation.
 */
public class HopDagPatternMatcher {
	static final HashSet<String> DEBUG_PATTERNS;
	static {
		// DEBUG_PATTERNS = new HashSet<>();
		// DEBUG_PATTERNS.add("batchNormdX");
		DEBUG_PATTERNS = null;
	}
	
	// Predicates for the current HOP
	List<HopPredicate> _predicates = new ArrayList<>();
	// Child matchers
	List<HopDagPatternMatcher> _children = new ArrayList<>();
	private boolean _isLeaf = false;
	
	static boolean DEBUG_REWRITES = false; // This is set by HopPatternRewriter. Please use DEBUG_PATTERNS instead.
	
	// Simple utility for debugging the rewrites
	public static class HopPredicate implements Predicate<Hop> {
		final String _name;
		final Function<Hop, Boolean> _pred;
		public HopPredicate(String name, Function<Hop, Boolean> pred) {
			_name = name;
			_pred = pred;
		}
		@Override
		public boolean test(Hop h) {
			return _pred.apply(h);
		}
		@Override
	    public String toString() {
	        return _name;
	    }
	}
	
	/**
	 * Adds a predicate to the pattern matcher
	 * 
	 * @param name name of the pattern for debugging
	 * @param pred higher order function that takes as an input a hop and returns true if the pattern matches else false
	 * @return this
	 */
	public HopDagPatternMatcher addPredicate(String name, Function<Hop, Boolean> pred) {
		_predicates.add(new HopPredicate(name, pred));
		return this;
	}
	
	/**
	 * Add child pattern matcher
	 * @param children list of childer
	 * @return this
	 */
	public HopDagPatternMatcher addChildMatcher(HopDagPatternMatcher... children) {
		for(int i = 0; i < children.length; i++) {
			_children.add(children[i]);
		}
		return this;
	}
	
	/**
	 * Get the matched HOP DAGs
	 * @param varName variable names
	 * @return matched HOP
	 */
	public Hop getMatchedHop(String varName) {
		
		if(matchedHops == null || !matchedHops.containsKey(varName)) {
			throw new RuntimeException("Incorrect usage: the variable " + varName + " is not registered as input.");
		}
		return matchedHops.get(varName);
	}
	
	/**
	 * Return the value 
	 * 
	 * @param varName variable name
	 * @return the value of the LiteralOp 
	 */
	public double getLiteralValue(String varName) {
		return OptimizerUtils.rEvalSimpleDoubleExpression(getMatchedHop(varName), new HashMap<>());
	}
	
	@Override
    public String toString() {
        return _predicates.size() >= 1 ? _predicates.get(0).toString() : "";
    }
	
	/**
	 * Match the given HOP DAG
	 * 
	 * @param h root node of the HOP DAG 
	 * @return true if HOP DAG matches
	 */
	public boolean matches(Hop h) {
		visited.clear();
		matchedHops.clear();
		return matchHelper(this, h);
	}
	
	private HashMap<String, Hop> matchedHops = new HashMap<>();
	private String variableName;
	private HashMap<HopDagPatternMatcher, Hop> visited = new HashMap<>(); // Map of matched hops
	private boolean matchHelper(HopDagPatternMatcher root, Hop h) {
		if(h == null) {
			return false;
		}
		else if(_children.size() > 0 && h.getInput().size() < _children.size()) {
			if(DEBUG_REWRITES) {
				System.out.println("The expected number of children (" + _children.size() + ") didnot match the number of inputs (" + h.getInput().size() + ") " + this);
			}
			return false;
		}
		if(root.visited.containsKey(this)) {
			Hop h1 = root.visited.get(this);
			if(h == h1) {
				if(DEBUG_REWRITES)
					System.out.println("MATCHED: Early exit as the given HOP has been already matched by the matcher." + this); 
				return true; // Early exit as the given HOP has been already matched by the matcher
			}
			else if(_isLeaf) {
				if(h.getDataType() == h1.getDataType() && h.getDataType() == DataType.SCALAR) {
					return OptimizerUtils.rEvalSimpleDoubleExpression(h, new HashMap<>()) == OptimizerUtils.rEvalSimpleDoubleExpression(h1, new HashMap<>());
				}
				return false; // Mismatched or unknown datatypes or matched with different hops
			}
		}
		
		for(HopPredicate p : _predicates) {
			if(!p.test(h)) {
				if(DEBUG_REWRITES) {
					System.out.println("The predicate " + p.toString() + " failed.");
				}
				return false;
			}
		}
		int index = 0;
		for(HopDagPatternMatcher child : _children) {
			if(!child.matchHelper(root, h.getInput().get(index))) {
				return false;
			}
			index++;
		}
		if(_isLeaf) {
			root.matchedHops.put(variableName, h);
		}
		
		root.visited.put(this, h);
		if(DEBUG_REWRITES)
			System.out.println("MATCHED: " + this + " to " + Explain.explain(h));
		return true;
	}
	

	// Simple helper utilities for adding predicates
	private HopDagPatternMatcher isScalar() {
		return this.addPredicate("isScalar", h -> h.getDataType() == DataType.SCALAR);
	}
	private HopDagPatternMatcher isMatrix() {
		return this.addPredicate("isMatrix", h -> h.getDataType() == DataType.MATRIX);
	}
	public HopDagPatternMatcher fitsOnGPU(double constant) {
		return this.addPredicate("fitsOnGPU", h -> _fitsOnGPU(h, constant));
	}
	
	// Factory methods:
	public static HopDagPatternMatcher dummy = new HopDagPatternMatcher();
	public static HopDagPatternMatcher rowMeans(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("rowMeans", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.MEAN && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChildMatcher(child1);
	}
	public static HopDagPatternMatcher rowVars(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("rowVars", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.VAR && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChildMatcher(child1);
	}
	public static HopDagPatternMatcher colVars(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("colVars", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.VAR && ((AggUnaryOp)h).getDirection() == Direction.Col)
			.addChildMatcher(child1);
	}
	public static HopDagPatternMatcher leaf(String _variableName, DataType dt) {
		HopDagPatternMatcher ret = new HopDagPatternMatcher();
		ret._isLeaf = true;
		ret.variableName = _variableName;
		if(dt == DataType.MATRIX) {
			return ret.isMatrix();
		}
		else if(dt == DataType.SCALAR) {
			return ret.isScalar();
		}
		else if(dt == DataType.UNKNOWN) {
			return ret;
		}
		else {
			throw new DMLRuntimeException("Unsupported datatype in pattern matcher:" + dt.name());
		}
	}
	public static HopDagPatternMatcher rowSums(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("rowSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.SUM && ((AggUnaryOp)h).getDirection() == Direction.Row)
			.addChildMatcher(child1);
	}
	public static HopDagPatternMatcher colSums(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("colSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.SUM && ((AggUnaryOp)h).getDirection() == Direction.Col)
			.addChildMatcher(child1);
	}
	public static HopDagPatternMatcher colMeans(HopDagPatternMatcher child1) {
		return new HopDagPatternMatcher().addPredicate("colSums", h -> 
			h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.MEAN && ((AggUnaryOp)h).getDirection() == Direction.Col)
			.addChildMatcher(child1);
	}
	public static HopDagPatternMatcher matrix(HopDagPatternMatcher X, HopDagPatternMatcher rows, HopDagPatternMatcher cols) {
		return new HopDagPatternMatcher().addPredicate("matrix_reshape", h -> HopRewriteUtils.isReorg(h, ReOrgOp.RESHAPE))
				.addChildMatcher(X, rows, cols);
	}
	public static HopDagPatternMatcher matrix(double X, HopDagPatternMatcher rows, HopDagPatternMatcher cols) {
		return new HopDagPatternMatcher().addPredicate("matrix_datagen", h -> HopRewriteUtils.isDataGenOpWithConstantValue(h, X))
				.addChildMatcher(rows, cols);
	}
	public static HopDagPatternMatcher matrix(double X, HopDagPatternMatcher rows, long cols) {
		return new HopDagPatternMatcher().addPredicate("matrix_datagen", h -> HopRewriteUtils.isDataGenOpWithConstantValue(h, X) && 
				h.getDim2() == cols)
				.addChildMatcher(rows, dummy);
	}
	public static HopDagPatternMatcher matrix(double X, long rows, HopDagPatternMatcher cols) {
		return new HopDagPatternMatcher().addPredicate("matrix_datagen", h -> HopRewriteUtils.isDataGenOpWithConstantValue(h, X) && 
				h.getDim1() == rows)
				.addChildMatcher(dummy, cols);
	}
	public static HopDagPatternMatcher matrix(double X, long rows, long cols) {
		return new HopDagPatternMatcher().addPredicate("matrix_datagen", h -> HopRewriteUtils.isDataGenOpWithConstantValue(h, X) && 
				h.getDim1() == rows && h.getDim2() == cols)
				.addChildMatcher(dummy, dummy);
	}
	public static HopDagPatternMatcher bias_add(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("bias_add", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASADD))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher bias_multiply(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("bias_multiply", h -> HopRewriteUtils.isDnn(h, OpOpDnn.BIASMULT))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher unaryMinus(HopDagPatternMatcher child) {
		return new HopDagPatternMatcher().addPredicate("unaryMinus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS)
				&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 0))
				.addChildMatcher(dummy, child);
	}
	public static HopDagPatternMatcher sqrt(HopDagPatternMatcher child) {
		return new HopDagPatternMatcher().addPredicate("sqrt", h -> HopRewriteUtils.isUnary(h, OpOp1.SQRT))
				.addChildMatcher(child);
	}
	public static HopDagPatternMatcher div(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher div(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static HopDagPatternMatcher div(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("div", h -> HopRewriteUtils.isBinary(h, OpOp2.DIV) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	
	public static HopDagPatternMatcher pow(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("pow", h -> HopRewriteUtils.isBinary(h, OpOp2.POW))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher pow(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("pow", h -> HopRewriteUtils.isBinary(h, OpOp2.POW) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	private static boolean matchDimensions(Hop h1, Hop h2) {
		return h1.getDim1() == h2.getDim1() && h1.getDim2() == h2.getDim2();
	}
	// This is used to differentiate between matrix-matrix and matrix-vector operations.
	public static HopDagPatternMatcher mm_plus(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS)
				&& matchDimensions(h.getInput().get(0), h.getInput().get(1)))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher plus(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher plus(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static HopDagPatternMatcher plus(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("plus", h -> HopRewriteUtils.isBinary(h, OpOp2.PLUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	public static HopDagPatternMatcher minus(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher minus(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static HopDagPatternMatcher minus(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("minus", h -> HopRewriteUtils.isBinary(h, OpOp2.MINUS) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	public static HopDagPatternMatcher mult(HopDagPatternMatcher child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT))
				.addChildMatcher(child1, child2);
	}
	public static HopDagPatternMatcher mult(double child1, HopDagPatternMatcher child2) {
		return new HopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), child1))
				.addChildMatcher(dummy, child2);
	}
	public static HopDagPatternMatcher mult(HopDagPatternMatcher child1, double child2) {
		return new HopDagPatternMatcher().addPredicate("mult", h -> HopRewriteUtils.isBinary(h, OpOp2.MULT) && 
				HopRewriteUtils.isLiteralOfValue(h.getInput().get(1), child2))
				.addChildMatcher(child1, dummy);
	}
	private static boolean _fitsOnGPU(Hop h, double multiplier) {
		double memEst = multiplier*h.getMemEstimate();
		return ConfigurationManager.isGPU() && h.dimsKnown() && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
}
