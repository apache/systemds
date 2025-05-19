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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.Compression.CompressConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.ForStatement;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.FunctionStatement;
import org.apache.sysds.parser.FunctionStatementBlock;
import org.apache.sysds.parser.IfStatement;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatement;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Rule: Compressed Re block if config compressed.linalg is enabled, we inject compression directions after read of
 * matrices if number of rows is above 1000 and cols at least 1.
 * 
 * In case of 'auto' compression, we apply compression if the data size is known to exceed aggregate cluster memory, the
 * matrix is used in loops, and all operations are supported over compressed matrices.
 */
public class RewriteCompressedReblock extends StatementBlockRewriteRule {
	private static final Log LOG = LogFactory.getLog(RewriteCompressedReblock.class.getName());

	private static final String TMP_PREFIX = "__cmtx";

	@Override
	public boolean createsSplitDag() {
		return false;
	}

	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus sate) {
		// check for inapplicable statement blocks
		if(!HopRewriteUtils.isLastLevelStatementBlock(sb) || sb.getHops() == null)
			return Arrays.asList(sb);

		// parse compression config
		final CompressConfig compress = ConfigurationManager.getCompressConfig();
		
		// perform compressed reblock rewrite
		if(compress.isEnabled()) {
			Hop.resetVisitStatus(sb.getHops());
			for(Hop h : sb.getHops())
				injectCompressionDirective(h, compress, sb.getDMLProg());
			Hop.resetVisitStatus(sb.getHops());
		}
		return Arrays.asList(sb);
	}

	@Override
	public List<StatementBlock> rewriteStatementBlocks(List<StatementBlock> sbs, ProgramRewriteStatus sate) {
		return sbs;
	}

	private static void injectCompressionDirective(Hop hop, CompressConfig compress, DMLProgram prog) {
		if(hop.isVisited() || hop.requiresCompression())
			return;

		// recursion for inputs.
		for(Hop hi : hop.getInput())
			injectCompressionDirective(hi, compress, prog);
		
		// filter candidates.
		if((HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD, DataType.SCALAR))//
			|| HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD, OpOpData.TRANSIENTWRITE)//
			|| hop instanceof LiteralOp)
			return;

		// check for compression conditions
		switch(compress) {
			case TRUE:
				if(satisfiesCompressionCondition(hop))
					hop.setRequiresCompression();
				break;
			case AUTO:
				if(OptimizerUtils.isSparkExecutionMode() && satisfiesAutoCompressionCondition(hop, prog))
					hop.setRequiresCompression();
				break;
			case COST:
				if(satisfiesCostCompressionCondition(hop, prog))
					hop.setRequiresCompression();
				break;
			default:
				break;
		}

		if(satisfiesDeCompressionCondition(hop)) {
			hop.setRequiresDeCompression();
		}

		hop.setVisited();
	}

	public static boolean satisfiesSizeConstraintsForCompression(Hop hop) {
		if(hop.getDim2() >= 1) {
			final long x = hop.getDim1();
			final long y = hop.getDim2();
			final boolean ret = 
				// If the Cube of the number of rows is greater than multiplying the number of columns by 1024.
				y << 10 <= x * x
				// is very sparse and at least 100 rows.
				|| (hop.getSparsity() < 0.0001 && y > 100);
			return ret;
		}
		else if(hop.getDim1() >= 1){
			// known rows. but not cols;
			boolean ret = hop.getDim1() > 10000;
			return ret;
		}
		else{
			return true; // unknown dimensions lets always try.
		}
	}

	public static boolean satisfiesCompressionCondition(Hop hop) {
		boolean satisfies = false;
		if(satisfiesSizeConstraintsForCompression(hop)){
			satisfies |= HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD) && !hop.isScalar();
			satisfies |= HopRewriteUtils.isTransformEncode(hop);
		}
		return satisfies;
	}

	public static boolean satisfiesAggressiveCompressionCondition(Hop hop) {
		//size-independent conditions (robust against unknowns)
		boolean satisfies = false;
		//size-dependent conditions
		if(satisfiesSizeConstraintsForCompression(hop)) {
			//matrix (no vector) ctable
			satisfies |= HopRewriteUtils.isTernary(hop, OpOp3.CTABLE) 
				&& hop.getInput(0).getDataType().isMatrix() 
				&& hop.getInput(1).getDataType().isMatrix();
			satisfies |= HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD);
			satisfies |= HopRewriteUtils.isUnary(hop, OpOp1.ROUND, OpOp1.FLOOR, OpOp1.NOT, OpOp1.CEIL);
			satisfies |= HopRewriteUtils.isBinary(hop, OpOp2.EQUAL, OpOp2.NOTEQUAL, OpOp2.LESS,
				OpOp2.LESSEQUAL, OpOp2.GREATER, OpOp2.GREATEREQUAL, OpOp2.AND, OpOp2.OR, OpOp2.MODULUS);
			satisfies |= HopRewriteUtils.isTernary(hop, OpOp3.CTABLE);
			satisfies &= !hop.isScalar();
		}
		if(LOG.isDebugEnabled() && satisfies)
			LOG.debug("Operation Satisfies: " + hop);
		return satisfies;
	}

	private static boolean satisfiesDeCompressionCondition(Hop hop) {
		// TODO decompression Condition
		return false;
	}

	private static boolean outOfCore(Hop hop) {
		double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(hop);
		double cacheSize = SparkExecutionContext.getDataMemoryBudget(true, true);
		return matrixPSize > cacheSize;
	}

	private static boolean ultraSparse(Hop hop) {
		double sparsity = OptimizerUtils.getSparsity(hop);
		return sparsity < MatrixBlock.ULTRA_SPARSITY_TURN_POINT;
	}

	private static boolean satisfiesAutoCompressionCondition(Hop hop, DMLProgram prog) {
		// check for basic compression condition
		if(!(satisfiesCompressionCondition(hop) && hop.getMemEstimate() >= OptimizerUtils.getLocalMemBudget()))
			return false;

		// determine if all operations are supported over compressed matrices,
		// but conditionally only if all other conditions are met
		if(hop.dimsKnown(true) && outOfCore(hop) && !ultraSparse(hop)) {
			return analyseProgram(hop, prog).isValidAutoCompression();
		}

		return false;
	}

	private static boolean satisfiesCostCompressionCondition(Hop hop, DMLProgram prog) {
		boolean satisfies = true;
		satisfies &= satisfiesAggressiveCompressionCondition(hop);
		satisfies &= hop.dimsKnown(false);
		satisfies &= analyseProgram(hop, prog).isValidAggressiveCompression();
		return satisfies;

	}

	private static ProbeStatus analyseProgram(Hop hop, DMLProgram prog) {
		ProbeStatus status = new ProbeStatus(hop.getHopID(), prog);
		for(StatementBlock sb : prog.getStatementBlocks())
			status.rAnalyzeProgram(sb);
		return status;
	}

	private static class ProbeStatus {
		private final long startHopID;
		private final DMLProgram prog;

		private int numberCompressedOpsExecuted = 0;
		private int numberDecompressedOpsExecuted = 0;
		private int inefficientSupportedOpsExecuted = 0;
		// private int superEfficientSupportedOpsExecuted = 0;

		private boolean foundStart = false;
		private boolean usedInLoop = false;
		private boolean condUpdate = false;
		private boolean nonApplicable = false;

		private HashSet<String> procFn = new HashSet<>();
		private HashSet<String> compMtx = new HashSet<>();

		private ProbeStatus(long hopID, DMLProgram p) {
			startHopID = hopID;
			prog = p;
		}

		private ProbeStatus(ProbeStatus status) {
			startHopID = status.startHopID;
			prog = status.prog;
			foundStart = status.foundStart;
			usedInLoop = status.usedInLoop;
			condUpdate = status.condUpdate;
			nonApplicable = status.nonApplicable;
			procFn.addAll(status.procFn);
		}

		private void rAnalyzeProgram(StatementBlock sb) {
			if(sb instanceof FunctionStatementBlock) {
				FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
				FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
				for(StatementBlock csb : fstmt.getBody())
					rAnalyzeProgram(csb);
			}
			else if(sb instanceof WhileStatementBlock) {
				WhileStatementBlock wsb = (WhileStatementBlock) sb;
				WhileStatement wstmt = (WhileStatement) wsb.getStatement(0);
				for(StatementBlock csb : wstmt.getBody())
					rAnalyzeProgram(csb);
				if(wsb.variablesRead().containsAnyName(compMtx))
					usedInLoop = true;
			}
			else if(sb instanceof IfStatementBlock) {
				IfStatementBlock isb = (IfStatementBlock) sb;
				IfStatement istmt = (IfStatement) isb.getStatement(0);
				for(StatementBlock csb : istmt.getIfBody())
					rAnalyzeProgram(csb);
				for(StatementBlock csb : istmt.getElseBody())
					rAnalyzeProgram(csb);
				if(isb.variablesUpdated().containsAnyName(compMtx))
					condUpdate = true;
			}
			else if(sb instanceof ForStatementBlock) { // incl parfor
				ForStatementBlock fsb = (ForStatementBlock) sb;
				ForStatement fstmt = (ForStatement) fsb.getStatement(0);
				for(StatementBlock csb : fstmt.getBody())
					rAnalyzeProgram(csb);
				if(fsb.variablesRead().containsAnyName(compMtx))
					usedInLoop = true;
			}
			else if(sb.getHops() != null) { // generic (last-level)
				ArrayList<Hop> roots = sb.getHops();
				Hop.resetVisitStatus(roots);
				// process entire HOP DAG starting from the roots
				for(Hop root : roots)
					rAnalyzeHopDag(root);
				// remove temporary variables
				compMtx.removeIf(n -> n.startsWith(TMP_PREFIX));
				Hop.resetVisitStatus(roots);
			}
		}

		private void rAnalyzeHopDag(Hop current) {
			if(current.isVisited())
				return;

			// process children recursively
			for(Hop input : current.getInput())
				rAnalyzeHopDag(input);

			// handle source persistent read
			if(current.getHopID() == startHopID) {
				compMtx.add(getTmpName(current));
				foundStart = true;
			}

			// 1) handle transient reads and writes (name mapping)
			if(HopRewriteUtils.isData(current, OpOpData.TRANSIENTWRITE) &&
				compMtx.contains(getTmpName(current.getInput().get(0))))
				compMtx.add(current.getName());
			else if(HopRewriteUtils.isData(current, OpOpData.TRANSIENTREAD) && compMtx.contains(current.getName()))
				compMtx.add(getTmpName(current));
			// handle individual hops
			else if(hasCompressedInput(current)) {
				if(current instanceof FunctionOp)
					handleFunctionOps(current);
				else
					handleApplicableOps(current);
			}
			current.setVisited();
		}

		private boolean hasCompressedInput(Hop hop) {
			if(compMtx.isEmpty())
				return false;
			for(Hop input : hop.getInput())
				if(compMtx.contains(getTmpName(input)))
					return true;
			return false;
		}

		private static String getTmpName(Hop hop) {
			return TMP_PREFIX + hop.getHopID();
		}

		private boolean isCompressed(Hop hop) {
			return compMtx.contains(getTmpName(hop));
		}

		private void handleFunctionOps(Hop current) {
			// TODO handle of functions in a more fine-grained manner
			// to cover special cases multiple calls where compressed
			// inputs might occur for different input parameters

			FunctionOp fop = (FunctionOp) current;
			String fkey = fop.getFunctionKey();
			if(!procFn.contains(fkey)) {
				// memoization to avoid redundant analysis and recursive calls
				procFn.add(fkey);
				// map inputs to function inputs
				FunctionStatementBlock fsb = prog.getFunctionStatementBlock(fkey);
				FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
				ProbeStatus status2 = new ProbeStatus(this);
				for(int i = 0; i < fop.getInput().size(); i++)
					if(compMtx.contains(getTmpName(fop.getInput().get(i))))
						status2.compMtx.add(fstmt.getInputParams().get(i).getName());
				// analyze function and merge meta info
				status2.rAnalyzeProgram(fsb);
				foundStart |= status2.foundStart;
				usedInLoop |= status2.usedInLoop;
				condUpdate |= status2.condUpdate;
				nonApplicable |= status2.nonApplicable;
				numberCompressedOpsExecuted += status2.numberCompressedOpsExecuted;
				numberDecompressedOpsExecuted += status2.numberDecompressedOpsExecuted;
				// map function outputs to outputs
				String[] outputs = fop.getOutputVariableNames();
				for(int i = 0; i < outputs.length; i++)
					if(status2.compMtx.contains(fstmt.getOutputParams().get(i).getName()))
						compMtx.add(outputs[i]);
			}
		}

		private void handleApplicableOps(Hop current) {
			// Valid with uncompressed outputs
			boolean compUCOut = false;
			// // tsmm
			// compUCOut |= (current instanceof AggBinaryOp && current.getDim2() <= current.getBlocksize() &&
			// ((AggBinaryOp) current).checkTransposeSelf() == MMTSJType.LEFT);

			// // mvmm
			// compUCOut |= (current instanceof AggBinaryOp && (current.getDim1() == 1 || current.getDim2() == 1));
			// compUCOut |= (HopRewriteUtils.isTransposeOperation(current) && current.getParent().size() == 1 &&
			// current.getParent().get(0) instanceof AggBinaryOp &&
			// (current.getParent().get(0).getDim1() == 1 || current.getParent().get(0).getDim2() == 1));

			compUCOut |= (current instanceof AggBinaryOp);
			compUCOut |= HopRewriteUtils.isBinaryMatrixColVectorOperation(current);

			boolean isAggregate = HopRewriteUtils
				.isAggUnaryOp(current, AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX, AggOp.MEAN);

			// If the aggregation function is done row wise.
			if(isAggregate && current.getDim2() < 2 && current.getDim1() >= 1000)
				inefficientSupportedOpsExecuted++;

			compUCOut |= isAggregate;

			// Valid compressed
			boolean compCOut = false;

			// Compressed Output if the operation is Binary scalar
			compCOut |= HopRewriteUtils.isBinaryMatrixScalarOperation(current);
			compCOut |= HopRewriteUtils.isBinaryMatrixRowVectorOperation(current);

			// Compressed Output possible through overlapping matrix.if the operation is right Matrix Multiply
			compCOut |= (current instanceof AggBinaryOp) && isCompressed(current.getInput().get(0));
			compUCOut = compCOut ? false : compUCOut;

			// Compressed Output if the operation is column bind.
			compCOut |= HopRewriteUtils.isBinary(current, OpOp2.CBIND);

			boolean metaOp = HopRewriteUtils.isUnary(current, OpOp1.NROW, OpOp1.NCOL);
			boolean ctableOp = HopRewriteUtils.isTernary(current, OpOp3.CTABLE);

			if(ctableOp) {
				numberCompressedOpsExecuted += 4;
				compCOut = true;
			}

			boolean applicable = compUCOut || compCOut || metaOp;

			if(applicable)
				numberCompressedOpsExecuted++;
			else {
				LOG.warn("Decompession op: " + current);
				numberDecompressedOpsExecuted++;
			}

			nonApplicable |= !(applicable);

			if(compCOut)
				compMtx.add(getTmpName(current));
		}

		private boolean isValidAutoCompression() {
			return foundStart && usedInLoop && !condUpdate && !nonApplicable;
		}

		private boolean isValidAggressiveCompression() {
			if(LOG.isDebugEnabled())
				LOG.debug(this.toString());
			return (inefficientSupportedOpsExecuted < numberCompressedOpsExecuted) &&
				(usedInLoop || numberCompressedOpsExecuted > 3) && numberDecompressedOpsExecuted < 1;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("Compressed ProbeStatus : hopID =" + startHopID);
			sb.append("\n CLA Ops         : " + numberCompressedOpsExecuted);
			sb.append("\n Decompress Ops  : " + numberDecompressedOpsExecuted);
			sb.append("\n Inefficient Ops : " + inefficientSupportedOpsExecuted);
			sb.append("\n foundStart " + foundStart + " , inLoop :" + usedInLoop + " , condUpdate : " + condUpdate
				+ " , nonApplicable : " + nonApplicable);
			sb.append("\n compressed Matrix: " + compMtx);
			sb.append("\n Prog Fn " + procFn);
			return sb.toString();
		}

	}
}
