/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.lineage;

import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.hops.AggBinaryOp;
import org.tugraz.sysds.hops.BinaryOp;
import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.Hop.OpOp2;
import org.tugraz.sysds.hops.Hop.OpOpN;
import org.tugraz.sysds.hops.IndexingOp;
import org.tugraz.sysds.hops.LiteralOp;
import org.tugraz.sysds.hops.NaryOp;
import org.tugraz.sysds.hops.ReorgOp;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.hops.rewrite.HopRewriteUtils;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.BasicProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.meta.MetaData;
import org.tugraz.sysds.utils.Explain;
import org.tugraz.sysds.utils.Explain.ExplainType;

public class RewriteCPlans
{
	private static final String LR_VAR = "__lrwrt";
	private static BasicProgramBlock _lrPB = null;
	private static ExecutionContext _lrEC = null;
	private static final Log LOG = LogFactory.getLog(RewriteCPlans.class.getName());
	
	public static boolean executeRewrites (Instruction curr, ExecutionContext ec)
	{
		boolean oneappend = false;
		boolean twoappend = false;
		MatrixBlock lastResult = null;
		if (LineageCache.isReusable(curr))
		{
			// If the input to tsmm came from cbind, look for both the inputs in cache.
			LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
			LineageItem item = items[0];

			// TODO restructuring of rewrites to make them all 
			// independent of each other and this opening condition here
			for (LineageItem source : item.getInputs())
				if (source.getOpcode().equalsIgnoreCase("append")) {
					for (LineageItem input : source.getInputs()) {
						// create tsmm lineage on top of the input of last append
						LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {input});
						if (LineageCache.probe(tmp)) {
							oneappend = true; // at least one entry to reuse
							if (lastResult == null)
								lastResult = LineageCache.get(curr, tmp);
						}
					}
					if (oneappend)
						break;   // no need to look for the next append

					// if not found in cache, look for two consecutive cbinds
					LineageItem input = source.getInputs()[0];
					if (input.getOpcode().equalsIgnoreCase("append")) {
						for (LineageItem L2appin : input.getInputs()) {
							LineageItem tmp = new LineageItem("comb", "append", new LineageItem[] {L2appin, source.getInputs()[1]});
							LineageItem toProbe = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {tmp});
							if (LineageCache.probe(toProbe)) {
								twoappend = true;
								if (lastResult == null)
									lastResult = LineageCache.get(curr, toProbe);
							}
						}
					}
				}
		}
		else
			return false;

		if (!oneappend && !twoappend) 
			return false;
		
		ExecutionContext lrwec = getExecutionContext();
		ExplainType et = DMLScript.EXPLAIN;
		// Disable explain not to print unnecessary logs
		// TODO extend recompiler to allow use without explain output
		DMLScript.EXPLAIN = ExplainType.NONE;

		try {
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			ArrayList<Instruction> newInst = oneappend ? rewriteCbindTsmm(curr, ec, lrwec, lastResult) : 
					twoappend ? rewrite2CbindTsmm(curr, ec, lrwec, lastResult) : null;
			if (DMLScript.STATISTICS) {
				LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
				LineageCacheStatistics.incrementPRewrites();
			}
			//execute instructions
			BasicProgramBlock pb = getProgramBlock();
			pb.setInstructions(newInst);
			LineageCacheConfig.shutdownReuse();
			pb.execute(lrwec);
			LineageCacheConfig.restartReuse();
			ec.setVariable(((ComputationCPInstruction)curr).output.getName(), lrwec.getVariable(LR_VAR));
			// add this to cache
			LineageCache.put(curr, ec);
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Error evaluating instruction: " + curr.toString() , e);
		}
		DMLScript.EXPLAIN = et;
		return true;
	}

	private static ArrayList<Instruction> rewriteCbindTsmm(Instruction curr, ExecutionContext ec, ExecutionContext lrwec, MatrixBlock lastResult) 
	{
		// Create a transient read op over the last tsmm result
		MetaData md = new MetaData(lastResult.getDataCharacteristics());
		MatrixObject newmo = new MatrixObject(ValueType.FP64, "lastResult", md);
		newmo.acquireModify(lastResult);
		newmo.release();
		lrwec.setVariable("lastResult", newmo);
		DataOp lastRes = HopRewriteUtils.createTransientRead("lastResult", lastResult);
		// Create rightIndex op to find the last matrix and the appended column
		// TODO: For now assumption is that a single column is being appended in a loop
		//       Need to go down the lineage to find number of columns are being appended.
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		IndexingOp oldMatrix = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), 
				new LiteralOp(mo.getNumRows()), new LiteralOp(1), new LiteralOp(mo.getNumColumns()-1));
		IndexingOp lastCol = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), 
				new LiteralOp(mo.getNumRows()), new LiteralOp(mo.getNumColumns()), 
				new LiteralOp(mo.getNumColumns()));
		// cell topRight = t(oldMatrix) %*% lastCol
		ReorgOp tOldM = HopRewriteUtils.createTranspose(oldMatrix);
		AggBinaryOp topRight = HopRewriteUtils.createMatrixMultiply(tOldM, lastCol);
		// cell bottomLeft = t(lastCol) %*% oldMatrix
		ReorgOp tLastCol = HopRewriteUtils.createTranspose(lastCol);
		AggBinaryOp bottomLeft = HopRewriteUtils.createMatrixMultiply(tLastCol, oldMatrix);
		// bottomRight = t(lastCol) %*% lastCol
		AggBinaryOp bottomRight = HopRewriteUtils.createMatrixMultiply(tLastCol, lastCol);
		// rowOne = cbind(lastRes, topRight)
		BinaryOp rowOne = HopRewriteUtils.createBinary(lastRes, topRight, OpOp2.CBIND);
		// rowTwo = cbind(bottomLeft, bottomRight)
		BinaryOp rowTwo = HopRewriteUtils.createBinary(bottomLeft, bottomRight, OpOp2.CBIND);
		// rbind(rowOne, rowTwo)
		BinaryOp lrwHop= HopRewriteUtils.createBinary(rowOne, rowTwo, OpOp2.RBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);
		
		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteCbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}

	private static ArrayList<Instruction> rewrite2CbindTsmm(Instruction curr, ExecutionContext ec, ExecutionContext lrwec, MatrixBlock lastResult) 
	{
		// Create a transient read op over the last tsmm result
		MetaData md = new MetaData(lastResult.getDataCharacteristics());
		MatrixObject newmo = new MatrixObject(ValueType.FP64, "lastResult", md);
		newmo.acquireModify(lastResult);
		newmo.release();
		lrwec.setVariable("lastResult", newmo);
		DataOp lastRes = HopRewriteUtils.createTransientRead("lastResult", lastResult);
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		
		// pull out the newly added column(2nd last) from the input matrix
		IndexingOp lastCol = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), new LiteralOp(mo.getNumRows()), 
				new LiteralOp(mo.getNumColumns()-1), new LiteralOp(mo.getNumColumns()-1));
		// apply t(lastCol) on i/p matrix to get the result vectors.
		ReorgOp tlastCol = HopRewriteUtils.createTranspose(lastCol);
		AggBinaryOp newCol = HopRewriteUtils.createMatrixMultiply(tlastCol, newMatrix);
		ReorgOp tnewCol = HopRewriteUtils.createTranspose(newCol);

		// push the result row & column inside the cashed block as 2nd last row and col respectively.
		IndexingOp topLeft = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(1), new LiteralOp(newmo.getNumRows()-1), 
				new LiteralOp(1), new LiteralOp(newmo.getNumColumns()-1));
		IndexingOp topRight = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(1), new LiteralOp(newmo.getNumRows()-1), 
				new LiteralOp(newmo.getNumColumns()), new LiteralOp(newmo.getNumColumns()));
		IndexingOp bottomLeft = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(newmo.getNumRows()), 
				new LiteralOp(newmo.getNumRows()), new LiteralOp(1), new LiteralOp(newmo.getNumColumns()-1));
		IndexingOp bottomRight = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(newmo.getNumRows()), 
				new LiteralOp(newmo.getNumRows()), new LiteralOp(newmo.getNumColumns()), new LiteralOp(newmo.getNumColumns()));
		IndexingOp topCol = HopRewriteUtils.createIndexingOp(tnewCol, new LiteralOp(1), new LiteralOp(mo.getNumColumns()-2), 
				new LiteralOp(1), new LiteralOp(1));
		IndexingOp bottomCol = HopRewriteUtils.createIndexingOp(tnewCol, new LiteralOp(mo.getNumColumns()), 
				new LiteralOp(mo.getNumColumns()), new LiteralOp(1), new LiteralOp(1));
		NaryOp rowOne = HopRewriteUtils.createNary(OpOpN.CBIND, topLeft, topCol, topRight);
		NaryOp rowTwo = HopRewriteUtils.createNary(OpOpN.CBIND, bottomLeft, bottomCol, bottomRight);
		NaryOp lrwHop = HopRewriteUtils.createNary(OpOpN.RBIND, rowOne, newCol, rowTwo);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewrite2CbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}
	
	private static ArrayList<Instruction> genInst(Hop hops, ExecutionContext ec) {
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(hops, ec.getVariables(), null, true, true, 0);
		LOG.debug("EXPLAIN LINEAGE REWRITE \nGENERIC (line "+hops.getBeginLine()+"):\n" + Explain.explain(hops,1));
		LOG.debug("EXPLAIN LINEAGE REWRITE \nGENERIC (line "+hops.getBeginLine()+"):\n" + Explain.explain(newInst,1));
		return newInst;
	}
	
	private static ExecutionContext getExecutionContext() {
		if( _lrEC == null )
			_lrEC = ExecutionContextFactory.createContext();
		return _lrEC;
	}

	private static BasicProgramBlock getProgramBlock() {
		if( _lrPB == null )
			_lrPB = new BasicProgramBlock( new Program() );
		return _lrPB;
	}
}
