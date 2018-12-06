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
import java.util.function.Function;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.OpOpDnn;
import static org.apache.sysml.hops.rewrite.HopDagPatternMatcher.*;
import static org.apache.sysml.parser.Expression.DataType.MATRIX;
import static org.apache.sysml.parser.Expression.DataType.SCALAR;


/*
 * -------------------------------------------------------------------------
 * Design documentation for hop rewrite rules that use HopDagPatternMatcher:
 * -------------------------------------------------------------------------
 * 
 * Typical (but not all) hop rewrite rules have following structure:
 * 1. Rules are grouped together in different Java classes and added in org.apache.sysml.hops.rewrite.ProgramRewriter.
 * 
 * 2. Each rule class inherits from HopRewriteRule and implements rewriteHopDAG method. Other class of rewrite rules are StatementBlockRewriteRule and are not covered by this approach.
 * 
 * 3. The structure of rewriteHopDAG is common across HopRewriteRule subclasses and usually have following pattern:
 *  if(root of the given HOP DAG matches certain pattern) {
 *    HopRewriteUtils.rewireAllParentChildReferences(root, newRoot)
 *  }
 *  else root
 * 
 * 4. To avoid redundancy, the above logic is implemented in the abstract class HopRewriteRuleWithPatternMatcher:
 *  ArrayList<HopPatternRewriter> patternRewriters =  getPatternRewriter();
 *    for(HopPatternRewriter patternRewriter : patternRewriters) {
 *      hi = patternRewriter.rewrite(hi);
 *  }
 * 
 * 5. The developer has to inherit from HopRewriteRuleWithPatternMatcher that implements the above logic
 * and write code for getPatternRewriter() that returns ArrayList<HopPatternRewriter>  
 * 
 * 6. Since the HOP pattern donot change during execution, it is convenient to implement them into a static variable: 
 * ArrayList<HopPatternRewriter> _rewriters
 * 
 * 7. The replacement part in each entry of patternMatcher invokes the helper methods in HopRewriteUtils to create a newRoot. For example: HopRewriteUtils.createDnnOp
 * 
 * 8. The below DSL is more readable if implemented with Scala's operator overloading, but it adds an dependency on scala library 
 * (in specific, scala uses scala.Function1 for implementing operator overloading).
 * Hence, to minimize the dependency, the DSL is implemented using static methods in HopDagPatternMatcher class.
 * We can revisit this if we plan to add scala as hard dependency in SystemML. 
 * 
 * 9. The matcher part in each entry of patternMatcher uses the DSL implemented in HopDagPatternMatcher to improve readability.
 * - The DSL mentioned above follows DML syntax that makes it convenient for an external contributer to understand and modify the HOP rewrites.
 * - It is important to note that the developer has to add the same scoping rules as SystemML.
 * - To create a newRoot HOP, it is important to have a mechanism to extract leaves of the matched pattern. This is implemented
 * by using leaf() method.
 * - Often, it is important to create a new HOP only if it it can fit into memory. For GPU, one can use the fitsOnGPU(multiplier) helper method.
 * 
 */
public class RewriteGPUSpecificOps extends HopRewriteRuleWithPatternMatcher {
	// -------------------------------------------------------------------------------------------
	
	private static HopDagPatternMatcher util_channel_sums(HopDagPatternMatcher X, HopDagPatternMatcher C, HopDagPatternMatcher HW) {
		// rowSums(matrix(colSums(X), rows=C, cols=HW))
		return rowSums(matrix(	colSums(X), C, HW));
	}
	
	// Pattern 1:
	private static final HopDagPatternMatcher _batchNormdX;
	static {
		HopDagPatternMatcher C = leaf("C",  SCALAR);
		HopDagPatternMatcher HW = leaf("HW",  SCALAR);
		HopDagPatternMatcher CHW = leaf("CHW",  SCALAR);
		HopDagPatternMatcher cache_inv_var = leaf("cache_inv_var", MATRIX);
		HopDagPatternMatcher dout = leaf("dout", MATRIX);
		HopDagPatternMatcher gamma = leaf("gamma", MATRIX);
		HopDagPatternMatcher X = leaf("X", MATRIX);
		HopDagPatternMatcher mean = leaf("mean", MATRIX);
		
		HopDagPatternMatcher centered = bias_add(X, unaryMinus(mean));
		
		// dnorm = bias_multiply(dout, gamma)  # shape (N, C*Hin*Win)
		HopDagPatternMatcher dnorm = bias_multiply(dout, gamma);
		// dmean_norm_branch = util::channel_sums(bias_multiply(dnorm, -cache_inv_var), C, Hin, Win)
		HopDagPatternMatcher dmean_norm_branch = util_channel_sums(bias_multiply(dnorm, unaryMinus(cache_inv_var)), C, HW) ;
		// dvar = util::channel_sums((-1/2) * bias_multiply(centered, cache_inv_var^3) * dnorm,
		//      C, Hin, Win)  # shape (C, 1)
		HopDagPatternMatcher dvar = util_channel_sums(mult(mult(-0.5, bias_multiply(centered, pow(cache_inv_var, 3))), dnorm),  C, HW);
		// dmean_var_branch = util::channel_sums((-2*oneByN*oneByHW) * centered, C, Hin, Win) * dvar
		HopDagPatternMatcher dmean_var_branch =
			mult(util_channel_sums(mult(leaf("const3", SCALAR), centered), C, HW), dvar);
		// dX_norm_branch = bias_multiply(dnorm, cache_inv_var)
		HopDagPatternMatcher dX_norm_branch = bias_multiply(dnorm, cache_inv_var);
		// dX_mean_branch = (oneByN*oneByHW) * bias_add(matrix(0, rows=1, cols=C*Hin*Win), dmean)
		HopDagPatternMatcher dX_mean_branch = mult(leaf("const1", SCALAR), bias_add(matrix(0, 1, CHW), 
				plus(dmean_norm_branch, dmean_var_branch) ));
		// dX_var_branch = (2*oneByN*oneByHW) * bias_multiply(centered, dvar)
		HopDagPatternMatcher dX_var_branch = mult(leaf("const2", SCALAR), bias_multiply(centered, dvar));
		_batchNormdX = plus(plus(dX_norm_branch, dX_mean_branch), dX_var_branch).fitsOnGPU(2);
	}
	private static final Function<Hop, Hop> _batchNormdXReplacer = hi -> {
		// double CHW = _batchNormdX.getLiteralValue("CHW");
		double HW = _batchNormdX.getLiteralValue("HW");
		double C = _batchNormdX.getLiteralValue("C");
		double const1 = _batchNormdX.getLiteralValue("const1"); // (oneByN*oneByHW)
		double const2 = _batchNormdX.getLiteralValue("const2"); // (2*oneByN*oneByHW)
		double const3 = _batchNormdX.getLiteralValue("const3"); // (-2*oneByN*oneByHW)
		if(2*const1 == const2 && const3 == -const2 && 
			hasSameDimensions(_batchNormdX.getMatchedHop("gamma"), _batchNormdX.getMatchedHop("mean")) &&
			hasSameDimensions(_batchNormdX.getMatchedHop("gamma"), _batchNormdX.getMatchedHop("cache_inv_var")) &&
			_batchNormdX.getMatchedHop("X").getDim2() == C*HW &&
			checkDimensions(_batchNormdX.getMatchedHop("gamma"), (long)C, 1)) {
			LOG.debug("Applied batchNormdX rewrite.");
			Hop newHop = HopRewriteUtils.createDnnOp(_batchNormdX, OpOpDnn.BATCH_NORM2D_BACKWARD_DX, 
					"X", "dout", "gamma", "mean", "cache_inv_var");
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		else if(DEBUG_REWRITES) {
			System.out.println("Couldnot apply batchNormdX rewrite."); 
			System.out.println((2*const1) + " == " + const2 + " && " + const3 + "== -" + const2 
			+ " && " + hasSameDimensions(_batchNormdX.getMatchedHop("gamma"), _batchNormdX.getMatchedHop("mean")) +  " && " + 
			hasSameDimensions(_batchNormdX.getMatchedHop("gamma"), _batchNormdX.getMatchedHop("cache_inv_var")) + " && " +
			_batchNormdX.getMatchedHop("X").getDim2() + " == " + C + "*" + HW  + " && " +
			checkDimensions(_batchNormdX.getMatchedHop("gamma"), (long)C, 1));
		}
		return hi;
	};
	
	
	
	// Pattern 2:
	// subgrp_vars = matrix(colVars(X) * ((N-1)/N), rows=C, cols=Hin*Win)
	// var = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
	private static final HopDagPatternMatcher _batchNormUpdatedVar; 
	static {
		HopDagPatternMatcher subgrp_vars = 
			matrix( 
					mult(colVars(leaf("X", MATRIX).fitsOnGPU(2)), leaf("varConst1", SCALAR)), // colVars(X) * ((N-1)/N)
					leaf("C", SCALAR),  	// rows=C
					leaf("HW", SCALAR)); // cols=Hin*Win
		_batchNormUpdatedVar = 
			mm_plus( 
					rowMeans(subgrp_vars), 
					mult(rowVars(leaf("subgrp_means", MATRIX)),  leaf("varConst2", SCALAR))); // rowVars(subgrp_means)*varConst2
	}
	private static final Function<Hop, Hop> _batchNormUpdatedVarReplacer = hi -> {
		double HW = _batchNormUpdatedVar.getLiteralValue("HW");
		if(_batchNormUpdatedVar.getLiteralValue("varConst2") == ((HW-1)/HW)) {
			LOG.debug("Applied batchNormUpdatedVar rewrite.");
			Hop newHop = HopRewriteUtils.createDnnOp(_batchNormUpdatedVar, OpOpDnn.UPDATE_EMA_VAR, 
					// varConst1 => ((N-1)/N)
					"subgrp_means", "X", "C", "HW", "varConst1");
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		return hi;
	};
	
	// Avoids unnecessary intermediates:
	// mean = cache_mean
	// centered = bias_add(X, -mean)  # shape (N, C*Hin*Win)
	// norm = bias_multiply(centered, cache_inv_var)  # shape (N, C*Hin*Win)
	// # Compute gradients during training
	// dgamma = util::channel_sums(dout*norm, C, Hin, Win)
	private static final HopDagPatternMatcher _batchNormDGamma;
	static {
		_batchNormDGamma = util_channel_sums(
				mult(	leaf("dout", MATRIX).fitsOnGPU(3),
						bias_multiply(bias_add(leaf("X", MATRIX), unaryMinus(leaf("ema_mean", MATRIX))), 
				leaf("ema_var", MATRIX))), leaf("C", SCALAR), leaf("HW", SCALAR));
	}
	private static final Function<Hop, Hop> _batchNormDGammaReplacer = hi -> {
		LOG.debug("Applied batchNormDGamma rewrite.");
		Hop newHop = HopRewriteUtils.createDnnOp(_batchNormDGamma, OpOpDnn.BATCH_NORM2D_BACKWARD_DGAMMA, 
				"ema_mean", "dout", "X", "ema_var");
		return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
	};
		
	// Pattern 3:
	private static final HopDagPatternMatcher _batchNormTest;
	static {
		// norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
		HopDagPatternMatcher norm = 
			bias_multiply(
					bias_add(leaf("X", MATRIX), unaryMinus(leaf("mean", MATRIX))), // bias_add(X, -mean)
					inv_var(leaf("var", MATRIX), leaf("eps", SCALAR))); // 1/sqrt(var+eps)
		// hi = bias_add(bias_multiply(norm, gamma), beta)
		_batchNormTest = 
			bias_add(
					bias_multiply(norm, leaf("gamma", MATRIX)), 
					leaf("beta", MATRIX))
			.fitsOnGPU(3);
	}
	private static final Function<Hop, Hop> _batchNormTestReplacer = hi -> {
		LOG.debug("Applied batchNormTest rewrite.");
		Hop newHop = HopRewriteUtils.createDnnOp(_batchNormTest, OpOpDnn.BATCH_NORM2D_TEST, "X", "gamma", "beta", "mean", "var", "eps");
		return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
	};
	
	// Pattern 4:
	// rowSums(matrix(colSums(X), rows=C, cols=HW))
	private static final HopDagPatternMatcher _channelSums = util_channel_sums(leaf("X", MATRIX).fitsOnGPU(2), leaf("C", SCALAR), leaf("HW", SCALAR));;
	private static final Function<Hop, Hop> _channelSumsReplacer = hi -> {
		LOG.debug("Applied channelSums rewrite.");
		Hop newHop = HopRewriteUtils.createDnnOp(_channelSums, OpOpDnn.CHANNEL_SUMS, "X", "C", "HW");
		return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
	};
	
	// Pattern 5:
	// (X - mu*v_prev) + (1+mu)*v
	private static final HopDagPatternMatcher _updateNesterovX = 
		mm_plus(
				minus(	// X - mu*v_prev
						leaf("X", MATRIX), 
						mult(	// mu*v_prev
								leaf("mu", SCALAR), 
								leaf("v_prev", MATRIX))),
				mult(	// (1+mu)*v
						leaf("onePlusMu", SCALAR), 
						leaf("v", MATRIX)))						
		.fitsOnGPU(3);
	private static final Function<Hop, Hop> _updateNesterovXReplacer = hi -> {
		if((1+_updateNesterovX.getLiteralValue("mu")) == _updateNesterovX.getLiteralValue("onePlusMu")) {
			Hop X = _updateNesterovX.getMatchedHop("X");
			Hop v = _updateNesterovX.getMatchedHop("v");
			Hop v_prev = _updateNesterovX.getMatchedHop("v_prev");
			if(hasSameDimensions(X, v) && hasSameDimensions(X, v_prev)) {
				LOG.debug("Applied updateNesterovX rewrite.");
				Hop newHop = HopRewriteUtils.createDnnOp(_updateNesterovX, OpOpDnn.UPDATE_NESTEROV_X, "X", "v", "v_prev", "mu");
				return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
			}
		}
		return hi;
	};
	
	// Pattern 6:
	// matrix(colMeans(X), rows=C, cols=Hin*Win)
	// This avoids unnecessary copy by the reshape operator
	private static final HopDagPatternMatcher _reshapeColMeans = 
		matrix(
				colMeans(leaf("X", MATRIX).fitsOnGPU(2)), // colMeans(X)
				leaf("C", SCALAR), 
				leaf("HW", SCALAR));
	private static final Function<Hop, Hop> _reshapeColMeansReplacer = hi -> {
		LOG.debug("Applied reshapeColMeans rewrite.");
		Hop newHop = HopRewriteUtils.createDnnOp(_reshapeColMeans, OpOpDnn.RESHAPE_COLMEANS, "X", "C", "HW");
		return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
	};
	
	// Pattern 7:
	// mu*ema_mean + (1-mu)*mean
	private static final HopDagPatternMatcher _updateEMA = 
		mm_plus( 	
				mult(	// mu*ema_mean
						leaf("mu", SCALAR), 
						leaf("ema_mean", MATRIX)), 
				mult(	// (1-mu)*mean
						leaf("oneMinusMu", SCALAR), 
						leaf("mean", MATRIX)))
		.fitsOnGPU(3);
	private static final Function<Hop, Hop> _updateEMAReplacer = hi -> {
		if((1-_updateEMA.getLiteralValue("mu")) == _updateEMA.getLiteralValue("oneMinusMu")) {
			LOG.debug("Applied updateEMA rewrite.");
			Hop newHop = HopRewriteUtils.createDnnOp(_updateEMA, OpOpDnn.UPDATE_EMA, "ema_mean", "mean", "mu");
			return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
		}
		return hi;
	};
	
	// Pattern 8:
	// 1/sqrt(var+epsilon)
	private static final HopDagPatternMatcher _invVar = 
		div(1, 
				sqrt(	// var+epsilon
						plus(	leaf("var", MATRIX), 
								leaf("eps", SCALAR) )))
		.fitsOnGPU(2);
	private static final Function<Hop, Hop> _invVarReplacer = hi -> {
		LOG.debug("Applied computeInverseVariance rewrite.");
		Hop newHop = HopRewriteUtils.createDnnOp(_invVar, OpOpDnn.INV_VAR, "var", "eps");
		return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
	};
	
	
	private static ArrayList<HopPatternRewriter> _rewriters = null;
	public ArrayList<HopPatternRewriter> getPatternRewriter() {
		if(_rewriters == null) {
			ArrayList<HopPatternRewriter> rewriters = new ArrayList<>();
			rewriters.add(new HopPatternRewriter("batchNormdX", _batchNormdX, _batchNormdXReplacer));
			rewriters.add(new HopPatternRewriter("batchNormTest", _batchNormTest, _batchNormTestReplacer));
			rewriters.add(new HopPatternRewriter("batchNormUpdatedVar", _batchNormUpdatedVar, _batchNormUpdatedVarReplacer));
			// rewriters.add(new HopPatternRewriter("batchNormDGamma", _batchNormDGamma, _batchNormDGammaReplacer));
			rewriters.add(new HopPatternRewriter("channelSums", _channelSums, _channelSumsReplacer));
			rewriters.add(new HopPatternRewriter("updateNesterovX", _updateNesterovX, _updateNesterovXReplacer));
			rewriters.add(new HopPatternRewriter("reshapeColMeans", _reshapeColMeans, _reshapeColMeansReplacer));
			rewriters.add(new HopPatternRewriter("updateEMA", _updateEMA, _updateEMAReplacer));
			rewriters.add(new HopPatternRewriter("invVar", _invVar, _invVarReplacer));
			_rewriters = rewriters;
		}
		return _rewriters;
	}
	
	
	// -------------------------------------------------------------------------------------------
	
	private static boolean hasSameDimensions(Hop x, Hop y) {
		return x.dimsKnown() && y.dimsKnown() && (x.getDim1() == y.getDim1()) && (x.getDim2() == y.getDim2());
	}
	
	private static boolean checkDimensions(Hop x, long dim1, long dim2) {
		return x.dimsKnown() && (x.getDim1() == dim1) && (x.getDim2() == dim2);
	}
}
