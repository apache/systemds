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

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.lops.Compression.CompressConfig;
import org.apache.sysds.lops.MMTSJ.MMTSJType;
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
 * Rule: CompressedReblock: If config compressed.linalg is enabled, we
 * inject compression directions after pread of matrices w/ both dims &gt; 1
 * (i.e., multi-column matrices). In case of 'auto' compression, we apply
 * compression if the datasize is known to exceed aggregate cluster memory,
 * the matrix is used in loops, and all operations are supported over 
 * compressed matrices.
 */
public class RewriteCompressedReblock extends StatementBlockRewriteRule
{
	private static final String TMP_PREFIX = "__cmtx";
	
	@Override
	public boolean createsSplitDag() {
		return false;
	}
	
	@Override
	public List<StatementBlock> rewriteStatementBlock(StatementBlock sb, ProgramRewriteStatus sate)
	{
		//check for inapplicable statement blocks
		if( !HopRewriteUtils.isLastLevelStatementBlock(sb)
			|| sb.getHops() == null )
			return Arrays.asList(sb);
		
		//parse compression config
		DMLConfig conf = ConfigurationManager.getDMLConfig();
		CompressConfig compress = CompressConfig.valueOf(
			conf.getTextValue(DMLConfig.COMPRESSED_LINALG).toUpperCase());
		
		//perform compressed reblock rewrite
		if( compress.isEnabled() ) {
			Hop.resetVisitStatus(sb.getHops());
			for( Hop h : sb.getHops() ) 
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
		if( hop.isVisited() || hop.requiresCompression() )
			return;
		
		// recursively process children
		for( Hop hi : hop.getInput() )
			injectCompressionDirective(hi, compress, prog);
		// check for compression conditions
		if( compress == CompressConfig.TRUE && satisfiesCompressionCondition(hop) 
			|| compress == CompressConfig.AUTO && satisfiesAutoCompressionCondition(hop, prog) )
		{
			hop.setRequiresCompression(true);
		}
		
		hop.setVisited();
	}
	
	private static boolean satisfiesCompressionCondition(Hop hop) {
		return HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD)
			&& hop.getDim1() > 1 && hop.getDim2() > 1; //multi-column matrix
	}
	
	private static boolean satisfiesAutoCompressionCondition(Hop hop, DMLProgram prog) {
		//check for basic compression condition
		if( !(satisfiesCompressionCondition(hop) 
			&& hop.getMemEstimate() >= OptimizerUtils.getLocalMemBudget()
			&& OptimizerUtils.isSparkExecutionMode()) )
			return false;
		
		//determine if data size exceeds aggregate cluster storage memory
		double matrixPSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(
			hop.getDim1(), hop.getDim2(), hop.getBlocksize(), hop.getNnz());
		double cacheSize = SparkExecutionContext.getDataMemoryBudget(true, true);
		boolean outOfCore = matrixPSize > cacheSize;
		
		//determine if matrix is ultra sparse (and hence serialized)
		double sparsity = OptimizerUtils.getSparsity(hop.getDim1(), hop.getDim2(), hop.getNnz());
		boolean ultraSparse = sparsity < MatrixBlock.ULTRA_SPARSITY_TURN_POINT;
		
		//determine if all operations are supported over compressed matrices,
		//but conditionally only if all other conditions are met
		if( hop.dimsKnown(true) && outOfCore && !ultraSparse ) {
			//analyze program recursively, including called functions
			ProbeStatus status = new ProbeStatus(hop.getHopID(), prog);
			for( StatementBlock sb : prog.getStatementBlocks() )
				rAnalyzeProgram(sb, status);
			
			//applicable if used in loop (amortized compressed costs), 
			// no conditional updates in if-else branches
			// and all operations are applicable (no decompression costs)
			boolean ret = status.foundStart && status.usedInLoop 
				&& !status.condUpdate && !status.nonApplicable;
			if( LOG.isDebugEnabled() ) {
				LOG.debug("Auto compression: "+ret+" (dimsKnown="+hop.dimsKnown(true)
					+ ", outOfCore="+outOfCore+", !ultraSparse="+!ultraSparse
					+", foundStart="+status.foundStart+", usedInLoop="+status.foundStart
					+", !condUpdate="+!status.condUpdate+", !nonApplicable="+!status.nonApplicable+")");
			}
			return ret;
		}
		else if( LOG.isDebugEnabled() ) {
			LOG.debug("Auto compression: false (dimsKnown="+hop.dimsKnown(true)
				+ ", outOfCore="+outOfCore+", !ultraSparse="+!ultraSparse+")");
		}
		return false;
	}
	
	private static void rAnalyzeProgram(StatementBlock sb, ProbeStatus status) 
	{
		if(sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock csb : fstmt.getBody())
				rAnalyzeProgram(csb, status);
		}
		else if(sb instanceof WhileStatementBlock) {
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock csb : wstmt.getBody())
				rAnalyzeProgram(csb, status);
			if( wsb.variablesRead().containsAnyName(status.compMtx) )
				status.usedInLoop = true;
		}	
		else if(sb instanceof IfStatementBlock) {
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for (StatementBlock csb : istmt.getIfBody())
				rAnalyzeProgram(csb, status);
			for (StatementBlock csb : istmt.getElseBody())
				rAnalyzeProgram(csb, status);
			if( isb.variablesUpdated().containsAnyName(status.compMtx) )
				status.condUpdate = true;
		}
		else if(sb instanceof ForStatementBlock) { //incl parfor
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock csb : fstmt.getBody())
				rAnalyzeProgram(csb, status);
			if( fsb.variablesRead().containsAnyName(status.compMtx) )
				status.usedInLoop = true;
		}
		else if( sb.getHops() != null ) { //generic (last-level)
			ArrayList<Hop> roots = sb.getHops();
			Hop.resetVisitStatus(roots);
			//process entire HOP DAG starting from the roots
			for( Hop root : roots )
				rAnalyzeHopDag(root, status);
			//remove temporary variables
			status.compMtx.removeIf(n -> n.startsWith(TMP_PREFIX));
			Hop.resetVisitStatus(roots);
		}
	}
	
	private static void rAnalyzeHopDag(Hop current, ProbeStatus status) 
	{
		if( current.isVisited() )
			return;
		
		//process children recursively
		for( Hop input : current.getInput() )
			rAnalyzeHopDag(input, status);
		
		//handle source persistent read
		if( current.getHopID() == status.startHopID ) {
			status.compMtx.add(getTmpName(current));
			status.foundStart = true;
		}
		
		//handle individual hops
		//a) handle function calls
		if( current instanceof FunctionOp 
			&& hasCompressedInput(current, status) )
		{
			//TODO handle of functions in a more fine-grained manner
			//to cover special cases multiple calls where compressed
			//inputs might occur for different input parameters
			
			FunctionOp fop = (FunctionOp) current;
			String fkey = fop.getFunctionKey();
			if( !status.procFn.contains(fkey) ) {
				//memoization to avoid redundant analysis and recursive calls
				status.procFn.add(fkey);
				//map inputs to function inputs
				FunctionStatementBlock fsb = status.prog.getFunctionStatementBlock(fkey);
				FunctionStatement fstmt = (FunctionStatement) fsb.getStatement(0);
				ProbeStatus status2 = new ProbeStatus(status);
				for(int i=0; i<fop.getInput().size(); i++)
					if( status.compMtx.contains(getTmpName(fop.getInput().get(i))) )
						status2.compMtx.add(fstmt.getInputParams().get(i).getName());
				//analyze function and merge meta info
				rAnalyzeProgram(fsb, status2);
				status.foundStart |= status2.foundStart;
				status.usedInLoop |= status2.usedInLoop;
				status.condUpdate |= status2.condUpdate;
				status.nonApplicable |= status2.nonApplicable;
				//map function outputs to outputs
				String[] outputs = fop.getOutputVariableNames();
				for( int i=0; i<outputs.length; i++ )
					if( status2.compMtx.contains(fstmt.getOutputParams().get(i).getName()) )
						status.compMtx.add(outputs[i]);
			}
		}
		//b) handle transient reads and writes (name mapping)
		else if( HopRewriteUtils.isData(current, OpOpData.TRANSIENTWRITE)
			&& status.compMtx.contains(getTmpName(current.getInput().get(0))))
			status.compMtx.add(current.getName());
		else if( HopRewriteUtils.isData(current, OpOpData.TRANSIENTREAD)
		&& status.compMtx.contains(current.getName()) )
			status.compMtx.add(getTmpName(current));
		//c) handle applicable operations
		else if( hasCompressedInput(current, status) ) {
			boolean compUCOut = //valid with uncompressed outputs
				(current instanceof AggBinaryOp && current.getDim2()<= current.getBlocksize() //tsmm
					&& ((AggBinaryOp)current).checkTransposeSelf()==MMTSJType.LEFT)
				|| (current instanceof AggBinaryOp && (current.getDim1()==1 || current.getDim2()==1)) //mvmm
				|| (HopRewriteUtils.isTransposeOperation(current) && current.getParent().size()==1
					&& current.getParent().get(0) instanceof AggBinaryOp 
					&& (current.getParent().get(0).getDim1()==1 || current.getParent().get(0).getDim2()==1))
				|| HopRewriteUtils.isAggUnaryOp(current, AggOp.SUM, AggOp.SUM_SQ, AggOp.MIN, AggOp.MAX);
				// AggOp.SUM, Direction.Col
			boolean compCOut = //valid with compressed outputs
				HopRewriteUtils.isBinaryMatrixScalarOperation(current)
				|| HopRewriteUtils.isBinary(current, OpOp2.CBIND);
			boolean metaOp = HopRewriteUtils.isUnary(current, OpOp1.NROW, OpOp1.NCOL);
			status.nonApplicable |= !(compUCOut || compCOut || metaOp);
			if( compCOut )
				status.compMtx.add(getTmpName(current));
		}
		
		current.setVisited();
	}
	
	private static String getTmpName(Hop hop) {
		return TMP_PREFIX + hop.getHopID();
	}
	
	private static boolean hasCompressedInput(Hop hop, ProbeStatus status) {
		if( status.compMtx.isEmpty() )
			return false;
		for( Hop input : hop.getInput() )
			if( status.compMtx.contains(getTmpName(input)) )
				return true;
		return false;
	}
	
	private static class ProbeStatus {
		private final long startHopID;
		private final DMLProgram prog;
		private boolean foundStart = false;
		private boolean usedInLoop = false;
		private boolean condUpdate = false;
		private boolean nonApplicable = false;
		private HashSet<String> procFn = new HashSet<>();
		private HashSet<String> compMtx = new HashSet<>();
		public ProbeStatus(long hopID, DMLProgram p) {
			startHopID = hopID;
			prog = p;
		}
		public ProbeStatus(ProbeStatus status) {
			startHopID = status.startHopID;
			prog = status.prog;
			foundStart = status.foundStart;
			usedInLoop = status.usedInLoop;
			condUpdate = status.condUpdate;
			nonApplicable = status.nonApplicable;
			procFn.addAll(status.procFn);
		}
	}
}
