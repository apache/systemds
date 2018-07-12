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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp1;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.OpOpDnn;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.DnnOp;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;

/*
 * This class contains GPU-specific rewrites for following patterns:
 * 
 * 1. batchNormTest:
 * norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
 * hi = bias_add(bias_multiply(norm, gamma), beta)
 * 
 * 2. channelSum:
 * output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize))
 */
public class RewriteGPUSpecificOps extends HopRewriteRule {

	@Override
	public ArrayList<Hop> rewriteHopDAGs(ArrayList<Hop> roots, ProgramRewriteStatus state) {
		if( roots == null )
			return roots;

		//one pass rewrite-descend (rewrite created pattern)
		for( Hop h : roots )
			rule_GPUKernels( h, false );
		Hop.resetVisitStatus(roots, true);

		//one pass descend-rewrite (for rollup) 
		for( Hop h : roots )
			rule_GPUKernels( h, true );
		Hop.resetVisitStatus(roots, true);
		
		return roots;
	}

	@Override
	public Hop rewriteHopDAG(Hop root, ProgramRewriteStatus state) {
		if( root == null )
			return root;
		
		//one pass rewrite-descend (rewrite created pattern)
		rule_GPUKernels( root, false );
		
		root.resetVisitStatus();
		
		//one pass descend-rewrite (for rollup) 
		rule_GPUKernels( root, true );
		
		return root;
	}
	
	/**
	 * Fuse the kernel
	 * 
	 * @param hop high-level operator
	 * @param descendFirst true if recursively process children first
	 */
	private void rule_GPUKernels(Hop hop, boolean descendFirst) 
	{
		if(hop.isVisited())
			return;
		
		//recursively process children
		for( int i=0; i<hop.getInput().size(); i++)
		{
			Hop hi = hop.getInput().get(i);
			
			//process childs recursively first (to allow roll-up)
			if( descendFirst )
				rule_GPUKernels(hi, descendFirst); //see below
			
			hi = batchNormTest(hop, hi, i); 
			hi = channelSums(hop, hi, i); 
	
			if( !descendFirst )
				rule_GPUKernels(hi, descendFirst);
		}

		hop.setVisited();
	}
	
	private static boolean isBiasAdd(Hop h) {
		return h instanceof DnnOp && ((DnnOp) h).getOp() == OpOpDnn.BIASADD;
	}
	
	private static boolean isBiasMultiply(Hop h) {
		return h instanceof DnnOp && ((DnnOp) h).getOp() == OpOpDnn.BIASMULT;
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
	
	private static Hop getFirstInput(Hop h) {
		if(h == null || h.getInput() == null || h.getInput().size() < 1) {
			throw new RuntimeException("No input available for " + h);
		}
		return h.getInput().get(0);
	}
	
	private static Hop getSecondInput(Hop h) {
		if(h == null || h.getInput() == null || h.getInput().size() < 2) {
			throw new RuntimeException("No input available for " + h);
		}
		return h.getInput().get(1);
	}
	
	private static boolean isUnaryMinus(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.MINUS 
				&& Hop.computeSizeInformation(h.getInput().get(0)) == 0;
	}
	
	private static boolean isOneDivideBySqrt(Hop h) {
		return h instanceof BinaryOp && ((BinaryOp)h).getOp() == OpOp2.DIV 
				&& h.getInput().get(1) instanceof UnaryOp
				&& ((UnaryOp)h.getInput().get(1)).getOp() == OpOp1.SQRT
				&& Hop.computeSizeInformation(h.getInput().get(0)) == 1;
	}
	
	private static Hop channelSums(Hop parent, Hop hi, int pos) 
	{
		if(hi instanceof AggUnaryOp) {
			AggUnaryOp hop = (AggUnaryOp) hi;
			// output = rowSums(matrix(colSums(x), rows=numChannels, cols=imgSize*imgSize))
			if(hop.getOp() == AggOp.SUM && hop.getDirection() == Direction.Row
				&& hop.getInput().get(0) instanceof ReorgOp && ((ReorgOp)hop.getInput().get(0)).getOp() == ReOrgOp.RESHAPE) {
				Hop colSumsInput = hop.getInput().get(0).getInput().get(0);
				if(colSumsInput instanceof AggUnaryOp && ((AggUnaryOp)colSumsInput).getOp() == AggOp.SUM && ((AggUnaryOp)colSumsInput).getDirection() == Direction.Col) {
					ArrayList<Hop> inHops = new ArrayList<Hop>();
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
	
	private static Hop batchNormTest(Hop parent, Hop hi, int pos) 
	{		
		// norm = bias_multiply(bias_add(X, -mean), 1/sqrt(var+eps))
		// hi = bias_add(bias_multiply(norm, gamma), beta)
		// 2x for input and output and 1x for overhead
		if( isBiasAdd(hi) && isBiasMultiply(getFirstInput(hi)) && fitsOnGPU(hi, 3) ) {	
			Hop norm = getFirstInput(getFirstInput(hi));
			if(isBiasMultiply(norm) && isBiasAdd(getFirstInput(norm)) 
					&& isUnaryMinus(getSecondInput(getFirstInput(norm)))
					&& isOneDivideBySqrt(getSecondInput(norm))) {
				double eps = 0;
				Hop var = getFirstInput(getSecondInput(getSecondInput(norm)));
				if(var instanceof BinaryOp && ((BinaryOp) var).getOp() == OpOp2.PLUS &&
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
				Hop gamma = getSecondInput(getFirstInput(hi));
				Hop beta = getSecondInput(hi);
				ArrayList<Hop> inHops = new ArrayList<Hop>();
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
		}
		
		return hi;
	}

}
