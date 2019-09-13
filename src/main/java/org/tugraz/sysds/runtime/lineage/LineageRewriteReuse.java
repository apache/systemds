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
import org.tugraz.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.meta.MetaData;
import org.tugraz.sysds.utils.Explain;
import org.tugraz.sysds.utils.Explain.ExplainType;

public class LineageRewriteReuse
{
	private static final String LR_VAR = "__lrwrt";
	private static BasicProgramBlock _lrPB = null;
	private static ExecutionContext _lrEC = null;
	private static final Log LOG = LogFactory.getLog(LineageRewriteReuse.class.getName());
	
	public static boolean executeRewrites (Instruction curr, ExecutionContext ec)
	{
		ExecutionContext lrwec = getExecutionContext();
		ExplainType et = DMLScript.EXPLAIN;
		// Disable explain not to print unnecessary logs
		// TODO extend recompiler to allow use without explain output
		DMLScript.EXPLAIN = ExplainType.NONE;
		
		//check applicability and apply rewrite
		ArrayList<Instruction> newInst = rewriteTsmmCbind(curr, ec, lrwec);        //tsmm(cbind(X, deltaX)) using tsmm(X)
		newInst = (newInst == null) ? rewriteTsmm2Cbind(curr, ec, lrwec) : newInst;   //tsmm(cbind(cbind(X, deltaX), ones)) using tsmm(X)
		//tsmm(rbind(X, deltaX)) using C = tsmm(X) -> C + tsmm(deltaX)
		newInst = (newInst == null) ? rewriteTsmmRbind(curr, ec, lrwec) : newInst;
		//rbind(X,deltaX) %*% Y using C = X %*% Y -> rbind(C, deltaX %*% Y)
		newInst = (newInst == null) ? rewriteMatMulRbindLeft(curr, ec, lrwec) : newInst;
		//X %*% cbind(Y,deltaY)) using C = X %*% Y -> cbind(C, X %*% deltaY)
		newInst = (newInst == null) ? rewriteMatMulCbindRight(curr, ec, lrwec) : newInst;
		
		if (newInst == null)
			return false;
		
		//execute instructions & write the o/p to symbol table
		executeInst(newInst, lrwec);
		ec.setVariable(((ComputationCPInstruction)curr).output.getName(), lrwec.getVariable(LR_VAR));

		//put the result into the cache
		LineageCache.put(curr, ec);
		DMLScript.EXPLAIN = et;
		return true;
	}
	
	/*--------------------------------REWRITE METHODS------------------------------*/

	private static ArrayList<Instruction> rewriteTsmmCbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		MatrixBlock cachedEntry = isTsmmCbind(curr, ec);
		if (cachedEntry == null)
			return null;
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the cached tsmm result
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
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

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		
		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteCbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}
	
	private static ArrayList<Instruction> rewriteTsmmRbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		// Check the applicability of this rewrite.
		MatrixBlock cachedEntry = isTsmmRbind(curr, ec);
		if (cachedEntry == null)
			return null;
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last tsmm result
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		// Create rightIndex op to find the last appended rows 
		//TODO: support for block of rows
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		IndexingOp lastRow = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(mo.getNumRows()), 
				new LiteralOp(mo.getNumRows()), new LiteralOp(1), new LiteralOp(mo.getNumColumns()));
		// tsmm(X + lastRow) = tsmm(X) + tsmm(lastRow)
		ReorgOp tlastRow = HopRewriteUtils.createTranspose(lastRow);
		AggBinaryOp tsmm_lr = HopRewriteUtils.createMatrixMultiply(tlastRow, lastRow);
		BinaryOp lrwHop = HopRewriteUtils.createBinary(lastRes, tsmm_lr, OpOp2.PLUS);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		
		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteRbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}

	private static ArrayList<Instruction> rewriteTsmm2Cbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		MatrixBlock cachedEntry = isTsmm2Cbind(curr, ec);
		if (cachedEntry == null)
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last tsmm result
		MatrixObject newmo = convMBtoMO(cachedEntry);
		lrwec.setVariable("cachedEntry", newmo);
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
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

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}

		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewrite2CbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}

	private static ArrayList<Instruction> rewriteMatMulRbindLeft (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		MatrixBlock cachedEntry = isMatMulRbindLeft(curr, ec);
		if (cachedEntry == null)
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last ba+* result
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		// Create rightIndex op to find the last appended rows 
		//TODO: support for block of rows
		MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("leftMatrix", moL);
		DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
		MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
		lrwec.setVariable("rightMatrix", moR);
		DataOp rightMatrix = HopRewriteUtils.createTransientRead("rightMatrix", moR);
		//TODO avoid the indexing if possible (if deltaX found in cache)
		IndexingOp lastRow = HopRewriteUtils.createIndexingOp(leftMatrix, new LiteralOp(moL.getNumRows()), 
				new LiteralOp(moL.getNumRows()), new LiteralOp(1), new LiteralOp(moL.getNumColumns()));
		// ba+*(X+lastRow, Y) = rbind(ba+*(X, Y), ba+*(lastRow, Y))
		AggBinaryOp rowTwo = HopRewriteUtils.createMatrixMultiply(lastRow, rightMatrix);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.RBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		
		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteCbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}

	private static ArrayList<Instruction> rewriteMatMulCbindRight (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		MatrixBlock cachedEntry = isMatMulCbindRight(curr, ec);
		if (cachedEntry == null)
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last ba+* result
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		// Create rightIndex op to find the last appended column 
		//TODO: support for block of rows
		MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("leftMatrix", moL);
		DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
		MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
		lrwec.setVariable("rightMatrix", moR);
		DataOp rightMatrix = HopRewriteUtils.createTransientRead("rightMatrix", moR);
		//TODO avoid the indexing if possible (if deltaY found in cache)
		IndexingOp lastCol = HopRewriteUtils.createIndexingOp(rightMatrix, new LiteralOp(1), new LiteralOp(moR.getNumRows()), 
				new LiteralOp(moR.getNumColumns()), new LiteralOp(moR.getNumColumns()));
		// ba+*(X, Y+lastCol) = cbind(ba+*(X, Y), ba+*(X, lastCol))
		AggBinaryOp rowTwo = HopRewriteUtils.createMatrixMultiply(leftMatrix, lastCol);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.CBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		
		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteCbindTsmm APPLIED");
		return genInst(lrwWrite, lrwec);
	}
	
	/*------------------------REWRITE APPLICABILITY CHECKS-------------------------*/

	private static MatrixBlock isTsmmCbind(Instruction curr, ExecutionContext ec)
	{
		MatrixBlock cachedEntry = null;
		if (!LineageCache.isReusable(curr))
			return cachedEntry;

		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		LineageItem item = items[0];
		for (LineageItem source : item.getInputs())
			if (source.getOpcode().equalsIgnoreCase("cbind")) {
				for (LineageItem input : source.getInputs()) {
					// create tsmm lineage on top of the input of last append
					LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {input});
					if (LineageCache.probe(tmp)) {
						if (cachedEntry == null)
							cachedEntry = LineageCache.get(curr, tmp);
						}
					}
			}
		return cachedEntry;
	}

	private static MatrixBlock isTsmmRbind(Instruction curr, ExecutionContext ec)
	{
		MatrixBlock cachedEntry = null;
		if (!LineageCache.isReusable(curr))
			return cachedEntry;

		// If the input to tsmm came from rbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		LineageItem item = items[0];
		for (LineageItem source : item.getInputs())
			if (source.getOpcode().equalsIgnoreCase("rbind")) {
				for (LineageItem input : source.getInputs()) {
					// create tsmm lineage on top of the input of last append
					LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {input});
					if (LineageCache.probe(tmp)) {
						if (cachedEntry == null)
							cachedEntry = LineageCache.get(curr, tmp);
						}
					}
			}
		return cachedEntry;
	}
	
	private static MatrixBlock isTsmm2Cbind (Instruction curr, ExecutionContext ec)
	{
		MatrixBlock cachedEntry = null;
		if (!LineageCache.isReusable(curr))
			return cachedEntry;

		//TODO: support nary cbind
		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		LineageItem item = items[0];
		// look for two consecutive cbinds
		for (LineageItem source : item.getInputs())
			if (source.getOpcode().equalsIgnoreCase("cbind")) {
				LineageItem input = source.getInputs()[0];
				if (input.getOpcode().equalsIgnoreCase("cbind")) {
					for (LineageItem L2appin : input.getInputs()) {
						LineageItem tmp = new LineageItem("comb", "cbind", new LineageItem[] {L2appin, source.getInputs()[1]});
						LineageItem toProbe = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {tmp});
						if (LineageCache.probe(toProbe)) {
							if (cachedEntry == null)
								cachedEntry = LineageCache.get(curr, toProbe);
						}
					}
				}
			}
		return cachedEntry;
	}

	private static MatrixBlock isMatMulRbindLeft(Instruction curr, ExecutionContext ec)
	{
		MatrixBlock cachedEntry = null;
		if (!LineageCache.isReusable(curr))
			return cachedEntry;

		// If the left input to ba+* came from rbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("ba+*")) {
			LineageItem left= items[0].getInputs()[0];
			LineageItem right = items[0].getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase("rbind")){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of rbind = X
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {leftSource, right});
				if (LineageCache.probe(tmp))
					cachedEntry = LineageCache.get(curr, tmp);
			}
		}
		return cachedEntry;
	}

	private static MatrixBlock isMatMulCbindRight(Instruction curr, ExecutionContext ec)
	{
		MatrixBlock cachedEntry = null;
		if (!LineageCache.isReusable(curr))
			return cachedEntry;

		// If the right input to ba+* came from cbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("ba+*")) {
			LineageItem left = items[0].getInputs()[0];
			LineageItem right = items[0].getInputs()[1];
			if (right.getOpcode().equalsIgnoreCase("cbind")) {
				LineageItem rightSource = right.getInputs()[0]; //left inpur of rbind = X
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {left, rightSource});
				if (LineageCache.probe(tmp))
					cachedEntry = LineageCache.get(curr, tmp);
			}
		}
		return cachedEntry;
	}

	/*----------------------INSTRUCTIONS GENERATION & EXECUTION-----------------------*/

	private static ArrayList<Instruction> genInst(Hop hops, ExecutionContext ec) {
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(hops, ec.getVariables(), null, true, true, 0);
		LOG.debug("EXPLAIN LINEAGE REWRITE \nGENERIC (line "+hops.getBeginLine()+"):\n" + Explain.explain(hops,1));
		LOG.debug("EXPLAIN LINEAGE REWRITE \nGENERIC (line "+hops.getBeginLine()+"):\n" + Explain.explain(newInst,1));
		return newInst;
	}
	
	private static void executeInst (ArrayList<Instruction> newInst, ExecutionContext lrwec)
	{  
		// Disable explain not to print unnecessary logs
		// TODO extend recompiler to allow use without explain output
		DMLScript.EXPLAIN = ExplainType.NONE;

		try {
			//execute instructions
			BasicProgramBlock pb = getProgramBlock();
			pb.setInstructions(newInst);
			ReuseCacheType oldReuseOption = DMLScript.LINEAGE_REUSE;
			LineageCacheConfig.shutdownReuse();
			pb.execute(lrwec);
			LineageCacheConfig.restartReuse(oldReuseOption);
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Error executing lineage rewrites" , e);
		}
	}

	/*-------------------------------UTILITY METHODS----------------------------------*/
	
	private static MatrixObject convMBtoMO (MatrixBlock cachedEntry) {
		MetaData md = new MetaData(cachedEntry.getDataCharacteristics());
		MatrixObject mo = new MatrixObject(ValueType.FP64, "cachedEntry", md);
		mo.acquireModify(cachedEntry);
		mo.release();
		return mo;
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