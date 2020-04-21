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

package org.apache.sysds.hops.rewrite;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.DnnOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContextPool;

/*
 * This class contains GPU-specific rewrites for following patterns:
 * 
 * 1. batchNormTest: applied when mode="test" in batch normalization nn layer.
 * norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
 * hi = bias_add(bias_multiply(norm, gamma), beta)
 * 
 * 2. channelSum:
 * output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize))
 * 
 * 3. batchNormTrain: applied when mode="train" in batch normalization nn layer.
 * This rewrite is only enabled if none of the outputs are persistent writes as it assumes that 
 * FunctionOp will introduce a transient writes. This rewrite replaces the existing outputs of the matched pattern with transient reads.
 * 
 */
public class RewriteGPUSpecificOps extends HopRewriteRule {

	private static int _seq = 1;
	
	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for( int i = 0; i < roots.size(); i++ )
			rule_GPUKernels(roots, roots.get(i), false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup) 
		for( int i = 0; i < roots.size(); i++ )
			rule_GPUKernels(roots, roots.get(i), true );
		Hop.resetVisitStatus(roots, true);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return root;
		
		//one pass rewrite-descend (rewrite created pattern)
		rule_GPUKernels(null, root, false );
		
		root.resetVisitStatus();
		
		//one pass descend-rewrite (for rollup) 
		rule_GPUKernels(null, root, true );
		
		return root;
	}
	
	/**
	 * Fuse the kernel
	 * 
	 * @param roots root operators
	 * @param hop high-level operator
	 * @param descendFirst true if recursively process children first
	 */
	private void rule_GPUKernels(ArrayList<Hop> roots, Hop hop, boolean descendFirst) 
	{
		if(hop.isVisited())
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++) {
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_GPUKernels(roots, hi, descendFirst); //see below
			
			if(roots != null) {
				//hi = batchNormTrain(roots, hop, hi, i);
			}
			hi = batchNormTest(hop, hi, i); 
			hi = channelSums(hop, hi, i); 
			hi = updateNesterovX(hop, hi, i);
	
			if( !descendFirst )
				rule_GPUKernels(roots, hi, descendFirst);
		}

		hop.setVisited();
	}
	
	private static boolean isBiasAdd(Hop h) {
		return HopRewriteUtils.isDnn(h, OpOpDnn.BIASADD);
	}
	
	private static boolean isBiasMultiply(Hop h) {
		return HopRewriteUtils.isDnn(h, OpOpDnn.BIASMULT);
	}
	
	private static boolean fitsOnGPU(Hop h, double multiplier) {
		double memEst = multiplier*h.getMemEstimate();
		return DMLScript.USE_ACCELERATOR && h.dimsKnown() && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
	
	private static boolean fitsOnGPU(ArrayList<Hop> inputHops, boolean isFirstSameSizeAsOutput) {
		return fitsOnGPU(inputHops, isFirstSameSizeAsOutput, 0);
	}
	
	private static boolean fitsOnGPU(ArrayList<Hop> inputHops, boolean isFirstSameSizeAsOutput, long additionalBytes) {
		double memEst = additionalBytes;
		boolean isFirst = true;
		for(Hop h : inputHops) {
			double est = h.getMemEstimate();
			if(est == OptimizerUtils.INVALID_SIZE) {
				return false;
			}
			else if(isFirst && isFirstSameSizeAsOutput) {
				isFirst = false;
				memEst += 2*est;
			}
			else {
				memEst += est;
			}
		}
		return DMLScript.USE_ACCELERATOR && OptimizerUtils.isMemoryBasedOptLevel() &&
				memEst < OptimizerUtils.getLocalMemBudget() && memEst < GPUContextPool.initialGPUMemBudget();
	}
	
	private static boolean hasFirstInput(Hop h) {
		return !(h == null || h.getInput() == null || h.getInput().size() < 1);
	}
	
	private static Hop getFirstInput(Hop h) {
		if(h == null || h.getInput() == null || h.getInput().size() < 1) {
			throw new RuntimeException("No input available for " + h);
		}
		return h.getInput().get(0);
	}
	
	private static boolean hasSecondInput(Hop h) {
		return !(h == null || h.getInput() == null || h.getInput().size() < 2);
	}
	
	private static Hop getSecondInput(Hop h) {
		if(h == null || h.getInput() == null || h.getInput().size() < 2) {
			throw new RuntimeException("Expected atleast two inputs for " + h);
		}
		return h.getInput().get(1);
	}
	
	private static Hop getThirdInput(Hop h) {
		if(h == null || h.getInput() == null || h.getInput().size() < 3) {
			throw new RuntimeException("Expected atleast three inputs for " + h);
		}
		return h.getInput().get(2);
	}
	
	private static boolean isUnaryMinus(Hop h) {
		return HopRewriteUtils.isBinary(h, OpOp2.MINUS)
			&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 0);
	}
	
	private static boolean isOneDivideBySqrt(Hop h) {
		return HopRewriteUtils.isBinary(h, OpOp2.DIV)
			&& HopRewriteUtils.isUnary(h.getInput().get(1), OpOp1.SQRT)
			&& HopRewriteUtils.isLiteralOfValue(h.getInput().get(0), 1);
	}
	
	private static Hop channelSums(Hop parent, Hop hi, int pos) {
		if(hi instanceof AggUnaryOp) {
			AggUnaryOp hop = (AggUnaryOp) hi;
			// output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize))
			if( hop.getOp() == AggOp.SUM && hop.getDirection() == Direction.Row
				&& HopRewriteUtils.isReorg(hop.getInput().get(0), ReOrgOp.RESHAPE) ) {
				Hop colSumsInput = hop.getInput().get(0).getInput().get(0);
				if(colSumsInput instanceof AggUnaryOp && ((AggUnaryOp)colSumsInput).getOp() == AggOp.SUM && ((AggUnaryOp)colSumsInput).getDirection() == Direction.Col) {
					ArrayList<Hop> inHops = new ArrayList<>();
					inHops.add(colSumsInput.getInput().get(0));
					long numChannels = Hop.computeSizeInformation(hop.getInput().get(0).getInput().get(1));
					long HW = Hop.computeSizeInformation(hop.getInput().get(0).getInput().get(2));
					if(numChannels > 0 && HW > 0 && fitsOnGPU(inHops, false, numChannels*8)) {
						inHops.add(new LiteralOp(numChannels));
						inHops.add(new LiteralOp(HW));
						LOG.debug("Applied channelSums rewrite.");
						Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
								OpOpDnn.CHANNEL_SUMS, inHops);
						return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
					}
				}
			}
		}
		return hi;
	}
	
	private static boolean isRowMeans(Hop h) {
		return h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.MEAN && ((AggUnaryOp)h).getDirection() == Direction.Row; 
	}
	
	private static boolean isRowVars(Hop h) {
		return h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.VAR && ((AggUnaryOp)h).getDirection() == Direction.Row; 
	}
	
	private static boolean isRowVars(Hop h, Hop childHop) {
		return isRowVars(h) && getFirstInput(h) == childHop; 
	}
	
	private static boolean isColMeans(Hop h) {
		return h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.MEAN && ((AggUnaryOp)h).getDirection() == Direction.Col; 
	}
	
	private static boolean isColVars(Hop h) {
		return h instanceof AggUnaryOp && ((AggUnaryOp)h).getOp() == AggOp.VAR && ((AggUnaryOp)h).getDirection() == Direction.Col; 
	}
	
	private static boolean isReshape(Hop h) {
		return h instanceof ReorgOp && ((ReorgOp)h).getOp() == ReOrgOp.RESHAPE;
	}
	
	private static boolean isReshape(Hop h, long expectedRows, long expectedCols) {
		return h instanceof ReorgOp && ((ReorgOp)h).getOp() == ReOrgOp.RESHAPE &&
				Hop.computeSizeInformation(getSecondInput(h)) == expectedRows && 
				Hop.computeSizeInformation(getThirdInput(h)) == expectedCols;
	}
	
	private static boolean isBinaryAdd(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.PLUS;
	}
	
	private static boolean isBinaryMSAdd(Hop h, double expectedValue) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.PLUS 
				&& getFirstInput(h).getDataType() == DataType.MATRIX && getSecondInput(h).getDataType() == DataType.SCALAR
				&& OptimizerUtils.rEvalSimpleDoubleExpression(getSecondInput(h), new HashMap<>()) == expectedValue;
	}
	
	private static boolean isBinaryMMAdd(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.PLUS 
				&& getFirstInput(h).getDataType() == DataType.MATRIX && getSecondInput(h).getDataType() == DataType.MATRIX;
	}
	
	private static boolean isBinaryMMMinus(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MINUS 
				&& getFirstInput(h).getDataType() == DataType.MATRIX && getSecondInput(h).getDataType() == DataType.MATRIX;
	}
	
	private static boolean isBinaryMSMult(Hop h, double expectedValue) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MULT 
				&& getFirstInput(h).getDataType() == DataType.MATRIX && getSecondInput(h).getDataType() == DataType.SCALAR
				&& OptimizerUtils.rEvalSimpleDoubleExpression(getSecondInput(h), new HashMap<>()) == expectedValue;
	}
	
	private static boolean isBinarySSMinus(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MINUS 
				&& getFirstInput(h).getDataType() == DataType.SCALAR && getSecondInput(h).getDataType() == DataType.SCALAR;
	}
	
	private static boolean isBinarySSDiv(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.DIV 
				&& getFirstInput(h).getDataType() == DataType.SCALAR && getSecondInput(h).getDataType() == DataType.SCALAR;
	}
	
	private static boolean isBinarySMDiv(Hop h, double expectedValue) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.DIV 
				&& getFirstInput(h).getDataType() == DataType.SCALAR && getSecondInput(h).getDataType() == DataType.MATRIX 
				&& OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(h), new HashMap<>()) == expectedValue;
	}
	
	private static boolean isAnyBinaryAdd(ArrayList<Hop> hops) {
		if(hops != null) {
			for(Hop h : hops) {
				if(h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.PLUS)
					return true;
			}
		}
		return false;
	}
	
	private static boolean isBinaryMSMult(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MULT 
				&& getFirstInput(h).getDataType() == DataType.MATRIX && getSecondInput(h).getDataType() == DataType.SCALAR;
	}
	
	private static boolean isBinarySMMult(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MULT 
				&& getSecondInput(h).getDataType() == DataType.MATRIX && getFirstInput(h).getDataType() == DataType.SCALAR;
	}
	
	private static boolean isBinarySMMult(Hop h, double expectedVal) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MULT 
				&& getSecondInput(h).getDataType() == DataType.MATRIX && getFirstInput(h).getDataType() == DataType.SCALAR
				&& getValue(getFirstInput(h)) == expectedVal;
	}
	
	private static double getValue(Hop h) {
		return OptimizerUtils.rEvalSimpleDoubleExpression(h, new HashMap<>());
	}
	
	/**
	 * Checks if the "mean" hop is a moving average of mean in batch normalization layer.
	 *  
	 * @param mean hop to check against
	 * @param X input data
	 * @return true if the "mean" hop is a moving average of mean in batch normalization layer.
	 */
	private static boolean isBatchNormTrainMean(Hop mean, Hop X) {
		// subgrp_means = matrix(colMeans(X), rows=C, cols=Hin*Win)
		// mean = rowMeans(subgrp_means)
		return isRowMeans(mean) && isReshape(getFirstInput(mean)) && isColMeans(getFirstInput(getFirstInput(mean)))
				&& getFirstInput(getFirstInput(getFirstInput(mean))) == X;
	}
	
	/**
	 * Checks for nrow(X) pattern
	 * 
	 * @param expr hop to be matched
	 * @param X input X
	 * @return true if expr is nrow(X) else false
	 */
	private static boolean isNrowOfX(Hop expr, Hop X) {
		return expr instanceof UnaryOp && ((UnaryOp)expr).getOp() == OpOp1.NROW && getFirstInput(expr) == X;
	}
	
	/**
	 * Checks for the colVars(X) * ((N-1)/N) pattern
	 * 
	 * @param expr hop to be matched
	 * @param X input X
	 * @param ignoreCorrectionTerm whether to ignore the correction term ((N-1)/N).
	 * @return true if expr is colVars(X) * ((N-1)/N) else false
	 */
	private static boolean isCorrectedColVars(Hop expr, Hop X, boolean ignoreCorrectionTerm) {
		// colVars(X) * ((N-1)/N)
		if(isColVars(expr) && getFirstInput(expr) == X) {
			// Support no correction as well in this rewrite
			return true;
		}
		else if(X.rowsKnown()) {
			return isBinaryMSMult(expr, ((double)X.getDim1()-1)/X.getDim1()) && 
					isColVars(getFirstInput(expr)) && getFirstInput(getFirstInput(expr)) == X;
		}
		else if(isBinaryMSMult(expr) && 
				isColVars(getFirstInput(expr)) && getFirstInput(getFirstInput(expr)) == X) {
			if(ignoreCorrectionTerm) {
				return true;
			}
			Hop tmp = getSecondInput(expr);
			// ((N-1)/N)
			boolean isNMinus1Pattern = isBinarySSDiv(tmp) && isBinarySSMinus(getFirstInput(tmp)) &&
					getFirstInput(getFirstInput(tmp)) == getSecondInput(tmp) && 
					OptimizerUtils.rEvalSimpleDoubleExpression(getSecondInput(getFirstInput(tmp)), new HashMap<>()) == 1;
			boolean ret = isNMinus1Pattern && isNrowOfX(getSecondInput(tmp), X);
			if(LOG.isDebugEnabled()) {
				LOG.debug("Is the corrected column variance pattern for batch_norm_train rewrite when number of rows of X unknown matched:" + ret);
			}
			return ret;
		}
		return false;
	}
	
	/**
	 * Checks if the "var" hop is a moving average of variance in batch normalization layer.
	 *  
	 * @param mean previously matched mean hop
	 * @param var the hop to check against
	 * @param X input data hop
	 * @param subgrpMeans mean for subgroup mean
	 * @param ignoreCorrectionTerm whether to incore the correct term  (see isCorrectedColVars method in this class)
	 * @return true if the "var" hop is a moving average of variance in batch normalization layer.
	 */
	private static boolean isBatchNormTrainVar(Hop mean, Hop var, Hop  X, Hop subgrpMeans, boolean ignoreCorrectionTerm) {
		long numChannels = Hop.computeSizeInformation(getSecondInput(getFirstInput(mean)));
		long HW = Hop.computeSizeInformation(getThirdInput(getFirstInput(mean)));
		// subgrp_vars = matrix(colVars(X) * ((N-1)/N), rows=C, cols=Hin*Win)
		// var = rowMeans(subgrp_vars) + rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
		return numChannels > 0 && HW > 0 && isBinaryMMAdd(var) && isRowMeans(getFirstInput(var)) &&  
				// matrix(colVars(X) * ((N-1)/N), rows=C, cols=Hin*Win)
				isReshape(getFirstInput(getFirstInput(var)), numChannels, HW) &&
				isCorrectedColVars(getFirstInput(getFirstInput(getFirstInput(var))), X, ignoreCorrectionTerm) &&
				// rowVars(subgrp_means)*(((Hin*Win)-1)/(Hin*Win))
				isBinaryMSMult(getSecondInput(var), ((((double)HW)-1)/HW)) && 
				isRowVars(getFirstInput(getSecondInput(var)), subgrpMeans);
	}
	
	/**
	 * Checks and returns the matched hops for expression ema_mean_upd = mu*ema_mean + (1-mu)*mean  
	 * 
	 * @param rhsTimesOps hop representing BinaryOp of expression (1-mu)*mean 
	 * @param mu value of mu
	 * @return an array [ema_mean_upd, ema_mean] if expression matched, else null
	 */
	private static Hop [] getUpdatedMovingAverageExpressions(Hop rhsTimesOp, double mu) {
		if(rhsTimesOp == null || rhsTimesOp.getParent() == null || rhsTimesOp.getParent().size() != 1 || 
				!isBinarySMMult(rhsTimesOp) || !isBinaryAdd(rhsTimesOp.getParent().get(0)))
			return null;
		
		// Check (1-mu)*mean
		double expectedOneMinusMu = OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(rhsTimesOp), new HashMap<>());
		Hop plusOp = rhsTimesOp.getParent().get(0); 
		Hop lhsTimesOp = null;
		if(plusOp.getInput().get(0) == rhsTimesOp) {
			lhsTimesOp = plusOp.getInput().get(1); 
		}
		else {
			lhsTimesOp = plusOp.getInput().get(0);
		}
		
		if(expectedOneMinusMu == (1-mu) && plusOp.getParent() != null && plusOp.getParent().size() == 1 &&  
			isBinarySMMult(lhsTimesOp) && OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(lhsTimesOp), new HashMap<>()) == mu) {
			return new Hop[] {
				plusOp.getParent().get(0),
				getSecondInput(lhsTimesOp), 
				getSecondInput(rhsTimesOp)
			};
		}
		return null;
	}
	
	/**
	 * Checks (if exactly one of rhsTimesOps) and returns the matched hops for expression ema_mean_upd = mu*ema_mean + (1-mu)*mean  
	 * 
	 * @param rhsTimesOps array list of hop representing BinaryOp of expression (1-mu)*mean 
	 * @param mu value of mu
	 * @return an array [ema_mean_upd, ema_mean] if any of the expression matched, else null
	 */
	private static Hop [] getUpdatedMovingAverageExpressions(ArrayList<Hop> rhsTimesOps, double mu) {
		if(rhsTimesOps == null || rhsTimesOps.size() == 0)
			return null;
		
		Hop [] ret = null;
		for(Hop h : rhsTimesOps) {
			boolean matched = isUpdatedMovingAverageExpression(h, mu);
			if(matched && ret != null) {
				return null; // Multiple matches, cannot decide which one to fuse
			}
			else if(matched) {
				ret = getUpdatedMovingAverageExpressions(h, mu);
			}
		}
		
		return ret;
	}
	
	/**
	 * Checks and returns the mu in the expression ema_mean_upd = mu*ema_mean + (1-mu)*mean
	 * 
	 * @param rhsTimesOps hop representing BinaryOp of expression (1-mu)*mean
	 * @return value of mu if the expression matched else null 
	 */
	private static Double getMuFromUpdatedMovingAverageExpressions(ArrayList<Hop> rhsTimesOps) {
		if(rhsTimesOps == null || rhsTimesOps.size() == 0)
			return null;
		
		Double ret = null; 
		for(Hop h : rhsTimesOps) {
			boolean matched = isUpdatedMovingAverageExpression(h);
			if(matched && ret != null) {
				return null; // Multiple matches, cannot decide which one to fuse
			}
			else if(matched) {
				ret = -(OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(h), new HashMap<>())-1);
			}
		}
		return ret;
	}
	
	/**
	 * Checks for the expression ema_mean_upd = mu*ema_mean + (1-mu)*mean
	 * 
	 * @param rhsTimesOps hop representing BinaryOp of expression (1-mu)*mean
	 * @return true if expression matched
	 */
	private static boolean isUpdatedMovingAverageExpression(Hop rhsTimesOp) {
		if(rhsTimesOp == null || rhsTimesOp.getParent() == null || rhsTimesOp.getParent().size() != 1 || 
				!isBinarySMMult(rhsTimesOp) || !isBinaryAdd(rhsTimesOp.getParent().get(0)))
			return false;
		
		// Check (1-mu)*mean
		Hop plusOp = rhsTimesOp.getParent().get(0); 
		Hop lhsTimesOp = null;
		if(plusOp.getInput().get(0) == rhsTimesOp) {
			lhsTimesOp = plusOp.getInput().get(1); 
		}
		else {
			lhsTimesOp = plusOp.getInput().get(0);
		}
		
		if(plusOp.getParent() != null && plusOp.getParent().size() == 1 && isBinarySMMult(lhsTimesOp)) {
			return true;
		}
		return false;
	}
	
	// ema_mean_upd = mu*ema_mean + (1-mu)*mean
	// Returns true if expression matched, else false
	private static boolean isUpdatedMovingAverageExpression(Hop rhsTimesOp, double mu) {
		if(rhsTimesOp == null || rhsTimesOp.getParent() == null || rhsTimesOp.getParent().size() != 1 || 
				!isBinarySMMult(rhsTimesOp) || !isBinaryAdd(rhsTimesOp.getParent().get(0)))
			return false;
		
		// Check (1-mu)*mean
		double expectedOneMinusMu = OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(rhsTimesOp), new HashMap<>());
		Hop plusOp = rhsTimesOp.getParent().get(0); 
		Hop lhsTimesOp = null;
		if(plusOp.getInput().get(0) == rhsTimesOp) {
			lhsTimesOp = plusOp.getInput().get(1); 
		}
		else {
			lhsTimesOp = plusOp.getInput().get(0);
		}
		
		if(expectedOneMinusMu == (1-mu) && plusOp.getParent() != null && plusOp.getParent().size() == 1 &&  
			isBinarySMMult(lhsTimesOp) && OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(lhsTimesOp), new HashMap<>()) == mu) {
			return true;
		}
		return false;
	}
	
	/**
	 * Checks for the expression 1/sqrt(denom)
	 * 
	 * @param denom denominator of the expression to be matched
	 * @return true if the expression 1/sqrt(denom) matched else false
	 */
	private static boolean isOneBySqrt(Hop denom) {
		return denom.getParent() != null && denom.getParent().get(0) instanceof UnaryOp &&
				((UnaryOp)denom.getParent().get(0)).getOp() == OpOp1.SQRT &&
				denom.getParent().get(0).getParent() != null && denom.getParent().get(0).getParent().size() == 1 &&
				isBinarySMDiv(denom.getParent().get(0).getParent().get(0), 1);
	}
	
	/**
	 * Checks for the batch norm (mode="train") pattern using the helper isBatchNormTrainMean and isBatchNormTrainVar
	 * and returns a new FunctionOp if matched
	 * 
	 * @param roots root hops of the given statement block
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new FunctionOp or hi
	 */
	@SuppressWarnings("unused")
	private static Hop batchNormTrain(ArrayList<Hop> roots, Hop parent, Hop hi, int pos) 
	{		
		// norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
		// hi = bias_add(bias_multiply(norm, gamma), beta)
		// 2x for input and output and 1x for overhead
		// fitsOnGPU(hi, 3)
		if( hasFirstInput(hi) && isBiasAdd(hi) && isBiasMultiply(getFirstInput(hi)) ) {	
			Hop norm = getFirstInput(getFirstInput(hi));
			if(hasSecondInput(norm) && isBiasMultiply(norm) && isBiasAdd(getFirstInput(norm)) 
					&& hasSecondInput(getFirstInput(norm)) && isUnaryMinus(getSecondInput(getFirstInput(norm)))
					&& isOneDivideBySqrt(getSecondInput(norm))) {
				double eps = 0;
				Hop var = getFirstInput(getSecondInput(getSecondInput(norm)));
				if(isBinaryAdd(var) && (getFirstInput(var) instanceof LiteralOp || getSecondInput(var) instanceof LiteralOp)) {
					// eps + ema_var
					if(getFirstInput(var) instanceof LiteralOp) {
						eps = OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(var), new HashMap<>());
						var = getSecondInput(var);
					}
					else {
						eps = OptimizerUtils.rEvalSimpleDoubleExpression(getSecondInput(var), new HashMap<>());
						var = getFirstInput(var);
					}
				}
				// Generate batch norm test op
				Hop X = getFirstInput(getFirstInput(norm));
				Hop mean = getSecondInput(getSecondInput(getFirstInput(norm)));
				
				if(hasFirstInput(mean) && isBatchNormTrainMean(mean , X) && isBatchNormTrainVar(mean, var, X, getFirstInput(mean), false) &&
					mean.getParent() != null && mean.getParent().size() >= 2 && 
					var.getParent() != null && var.getParent().size() == 2) {
					Hop gamma = getSecondInput(getFirstInput(hi));
					Hop beta = getSecondInput(hi);
					
					// Always get mu from variance as it will have exactly one match of fusion pattern
					Double potentialMu = getMuFromUpdatedMovingAverageExpressions(var.getParent());
					if(potentialMu == null)
						return hi;
					double mu = potentialMu;
					
					Hop [] means = getUpdatedMovingAverageExpressions(mean.getParent(), mu);
					Hop [] vars = getUpdatedMovingAverageExpressions(var.getParent(), mu);
					if(means == null || vars == null)
						return hi;
					
					Hop varPlusEps = null;
					boolean isFirstBinaryAddOp = isAnyBinaryAdd(var.getParent().get(0).getParent());
                    boolean isSecondBinaryAddOp = isAnyBinaryAdd(var.getParent().get(1).getParent());
                    if(isFirstBinaryAddOp && !isSecondBinaryAddOp) {
                            varPlusEps = var.getParent().get(1);
                    }
                    else if(!isFirstBinaryAddOp && isSecondBinaryAddOp) {
                            varPlusEps = var.getParent().get(0);
                    }
					if(varPlusEps != null && isBinaryMSAdd(varPlusEps, eps) && isOneBySqrt(varPlusEps)) {
						
						Hop cache_var = varPlusEps.getParent().get(0).getParent().get(0);
						Hop ema_mean_upd = means[0];
						Hop ema_var_upd = vars[0];
						Hop ema_mean = means[1];
						Hop ema_var = vars[1];
						Hop cache_mean = means[2];
						
						
						ArrayList<Hop> inHops = new ArrayList<Hop>();
						inHops.add(X);
						inHops.add(gamma);
						inHops.add(beta);
						inHops.add(ema_mean);
						inHops.add(ema_var);
						inHops.add(new LiteralOp(eps));
						inHops.add(new LiteralOp(mu));
						Hop [] oldHops = {hi, ema_mean_upd, ema_var_upd, cache_mean, cache_var};
						
						// Since FunctionOp adds transientwrite explicitly, persistent writes are not supported
						if(!isAnyPersistentWrite(oldHops)) {
							LOG.debug("Applied batchNormTrain rewrite.");
							ArrayList<Hop> outputs = getMultiOutputHops(roots, oldHops);
							FunctionOp ret = new FunctionOp(FunctionType.MULTIRETURN_BUILTIN, DMLProgram.INTERNAL_NAMESPACE, "batch_norm2d_train", 
								null, inHops, outputs.stream().map(h -> h.getName()).toArray(String[]::new), outputs);
							Collections.reverse(roots);
							roots.add(ret);
							Collections.reverse(roots);
							return ret;
						}
					}
					
				}
			}
		}
		
		return hi;
	}
	
	// ------------------------------------------------------------
	/**
	 * Checks if any of the given output hop is a persistent write.
	 * 
	 * @param outputHops output hops to check
	 * @return true if any of the hop is a persistent write else false.
	 */
	private static boolean isAnyPersistentWrite(Hop [] outputHops) {
		for(Hop outHop : outputHops) {
			if(HopRewriteUtils.isData(outHop, OpOpData.PERSISTENTWRITE))
				return true;
		}
		return false;
	}
	
	/**
	 * Returns output hop for a multi-output FunctionOp to be created by rewrite.
	 * 
	 * @param roots root hops of statement block
	 * @param oldHops old output hops of the pattern
	 * @return new output hops that should be passed to FunctionOp
	 */
	private static ArrayList<Hop> getMultiOutputHops(ArrayList<Hop> roots, Hop [] oldHops) {
		ArrayList<Hop> ret = new ArrayList<>();
		for(int i = 0; i < oldHops.length; i++) {
			// Create a transient read as FunctionOp will add a transient write.
			if(HopRewriteUtils.isData(oldHops[i], OpOpData.PERSISTENTWRITE))
				throw new RuntimeException("Persistent write is not supported as output for the given rewrite." + oldHops[i]);
			// Generate a new name if the old output was not transient write.
			String name = HopRewriteUtils.isData(oldHops[i], OpOpData.TRANSIENTWRITE) ? oldHops[i].getName() : "_genGPU" + (_seq++);
			DataOp tRead = HopRewriteUtils.createTransientRead(name, oldHops[i]);
			HopRewriteUtils.rewireAllParentChildReferences(oldHops[i], tRead);
			ret.add(tRead);
			// Remove old output from roots to avoid unnecessary computation.
			if(roots.contains(oldHops[i])) {
				roots.remove(oldHops[i]);
			}
		}
		return ret;
	}
	// ------------------------------------------------------------
	
	/**
	 * Checks for the nesterov_update_x pattern (X = X - mu*v_prev + (1+mu)*v)
	 * and returns a new DnnOp if matched
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop updateNesterovX(Hop parent, Hop hi, int pos) {
		if(fitsOnGPU(hi, 4) && isBinaryMMAdd(hi) && isBinaryMMMinus(getFirstInput(hi))
			&& isBinarySMMult(getSecondInput(getFirstInput(hi))) 
			&& isBinarySMMult(getSecondInput(hi))) {
			Hop onePlusMu = getFirstInput(getSecondInput(hi));
			Hop tmp = getSecondInput(getFirstInput(hi));
			Hop mu = getFirstInput(tmp);
			if(isOnePlusMu(onePlusMu, mu)) {
				Hop v_prev = getSecondInput(tmp);
				Hop v = getSecondInput(getSecondInput(hi));
				Hop X = getFirstInput(getFirstInput(hi));
				if(hasSameDimensions(X, v) && hasSameDimensions(X, v_prev)) {
					ArrayList<Hop> inHops = new ArrayList<>();
					inHops.add(X);
					inHops.add(v);
					inHops.add(v_prev);
					inHops.add(mu);
					LOG.debug("Applied updateNesterovX rewrite.");
					Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
							OpOpDnn.UPDATE_NESTEROV_X, inHops);
					return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
				}
			}
		}
		return hi;
	}
	
	private static boolean hasSameDimensions(Hop x, Hop y) {
		return x.dimsKnown() && y.dimsKnown() && (x.getDim1() == y.getDim1()) && (x.getDim2() == y.getDim2());
	}
	
	private static boolean isOnePlusMu(Hop onePlusMu, Hop mu) {
		return (isBinarySMMult(onePlusMu, 1.0) && getSecondInput(onePlusMu) == mu) ||
				getValue(onePlusMu) == getValue(mu) + 1;
	}
	
	/**
	 * Checks for the batch norm (mode="test") pattern using the helper isBatchNormTrainMean and isBatchNormTrainVar
	 * and returns a new DnnOp if matched
	 * 
	 * @param parent parent of the input
	 * @param hi input to be matched
	 * @param pos position
	 * @return a new DnnOp or hi
	 */
	private static Hop batchNormTest(Hop parent, Hop hi, int pos) {
		// norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
		// hi = bias_add(bias_multiply(norm, gamma), beta)
		// 2x for input and output and 1x for overhead
		if(hasFirstInput(hi) && isBiasAdd(hi) && isBiasMultiply(getFirstInput(hi)) && fitsOnGPU(hi, 3) ) {
			Hop norm = getFirstInput(getFirstInput(hi));
			if(hasSecondInput(norm) && isBiasMultiply(norm) && isBiasAdd(getFirstInput(norm)) 
					&& isUnaryMinus(getSecondInput(getFirstInput(norm)))
					&& isOneDivideBySqrt(getSecondInput(norm))) {
				double eps = 0;
				Hop var = getFirstInput(getSecondInput(getSecondInput(norm)));
				if( HopRewriteUtils.isBinary(var, OpOp2.PLUS) &&
					(getFirstInput(var) instanceof LiteralOp || getSecondInput(var) instanceof LiteralOp)) {
					// eps + ema_var
					if(getFirstInput(var) instanceof LiteralOp) {
						eps = OptimizerUtils.rEvalSimpleDoubleExpression(getFirstInput(var), new HashMap<>());
						var = getSecondInput(var);
					}
					else {
						eps = OptimizerUtils.rEvalSimpleDoubleExpression(getSecondInput(var), new HashMap<>());
						var = getFirstInput(var);
					}
				}
				// Generate batch norm test op
				Hop X = getFirstInput(getFirstInput(norm));
				Hop mean = getSecondInput(getSecondInput(getFirstInput(norm)));
				
				// This guard disallows eager fusion of train batch normalization into test batch normalization
				boolean potentialForBatchNormTrain = !X.rowsKnown() && isBatchNormTrainMean(mean , X) && isBatchNormTrainVar(mean, var, X, getFirstInput(mean), true);
				if(!potentialForBatchNormTrain) {
					Hop gamma = getSecondInput(getFirstInput(hi));
					Hop beta = getSecondInput(hi);
					ArrayList<Hop> inHops = new ArrayList<>();
					inHops.add(X);
					inHops.add(gamma);
					inHops.add(beta);
					inHops.add(mean);
					inHops.add(var);
					inHops.add(new LiteralOp(eps));
					if(fitsOnGPU(inHops, true)) {
						LOG.debug("Applied batchNormTest rewrite.");
						Hop newHop = new DnnOp(hi.getName(), hi.getDataType(), hi.getValueType(),
								OpOpDnn.BATCH_NORM2D_TEST, inHops);
						return HopRewriteUtils.rewireAllParentChildReferences(hi, newHop);
					}
				}
				else {
					LOG.debug("Skipping batchNormTest rewrite as there is potential for batch normalization train rewrite after recompilation.");
				}
			}
		}
		
		return hi;
	}
}
