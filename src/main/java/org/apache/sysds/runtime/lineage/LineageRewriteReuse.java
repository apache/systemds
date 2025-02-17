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

package org.apache.sysds.runtime.lineage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionParser;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Explain.ExplainType;

public class LineageRewriteReuse
{
	private static final String LR_VAR = "__lrwrt";
	private static BasicProgramBlock _lrPB = null;
	private static ExecutionContext _lrEC = null;
	private static boolean _disableReuse = true;
	private static long _computeTime = 0;
	private static final Log LOG = LogFactory.getLog(LineageRewriteReuse.class.getName());
	
	public static boolean executeRewrites (Instruction curr, ExecutionContext ec)
	{
		ExecutionContext lrwec = getExecutionContext();
		ExplainType et = DMLScript.EXPLAIN;
		// Disable explain not to print unnecessary logs
		// TODO extend recompiler to allow use without explain output
		DMLScript.EXPLAIN = ExplainType.NONE;
		
		//check applicability and apply rewrite
		//tsmm(cbind(X, ones)) -> rbind(t(colSums(cbind(X, ones))[, 1:ncol-1]), colSums(cbind(X, ones)))
		ArrayList<Instruction> newInst = rewriteTsmmCbindOnes(curr, ec, lrwec);
		//tsmm(cbind(X, deltaX)) -> rbind(cbind(tsmm(X), t(X) %*% deltaX), cbind(t(deltaX) %*%X, tsmm(deltaX)))
		newInst = (newInst == null) ? rewriteTsmmCbind(curr, ec, lrwec) : newInst;
		//tsmm(cbind(cbind(X, deltaX), ones)) -> TODO
		newInst = (newInst == null) ? rewriteTsmm2Cbind(curr, ec, lrwec) : newInst;
		//tsmm(cbind(cbind(X, deltaX), ones)) -> TODO
		newInst = (newInst == null) ? rewriteTsmm2CbindSameLeft(curr, ec, lrwec) : newInst;
		//tsmm(rbind(X, deltaX)) -> tsmm(X) + tsmm(deltaX)
		newInst = (newInst == null) ? rewriteTsmmRbind(curr, ec, lrwec) : newInst;
		//rbind(X,deltaX) %*% Y -> rbind(X %*% Y, deltaX %*% Y)
		newInst = (newInst == null) ? rewriteMatMulRbindLeft(curr, ec, lrwec) : newInst;
		//X %*% cbind(Y,ones)) -> cbind(X %*% Y, rowSums(X))
		newInst = (newInst == null) ? rewriteMatMulCbindRightOnes(curr, ec, lrwec) : newInst;
		//X %*% cbind(Y,deltaY)) -> cbind(X %*% Y, X %*% deltaY)
		newInst = (newInst == null) ? rewriteMatMulCbindRight(curr, ec, lrwec) : newInst;
		//rbind(X, deltaX) * rbind(Y, deltaY) -> rbind(X * Y, deltaX * deltaY)
		newInst = (newInst == null) ? rewriteElementMulRbind(curr, ec, lrwec) : newInst;
		//cbind(X, deltaX) * cbind(Y, deltaY) -> cbind(X * Y, deltaX * deltaY)
		newInst = (newInst == null) ? rewriteElementMulCbind(curr, ec, lrwec) : newInst;
		//aggregate(target=cbind(X, deltaX,...) = cbind(aggregate(target=X,...), aggregate(target=deltaX,...)) for same agg function
		newInst = (newInst == null) ? rewriteAggregateCbind(curr, ec, lrwec) : newInst;
		//A %*% B[,1:k] = (A %*% B)[,1:k];
		newInst = (newInst == null) ? rewriteIndexingMatMul(curr, ec, lrwec) : newInst;
		//PCA --> lmDS pipeline
		newInst = (newInst == null) ? rewritePcaTsmm(curr, ec, lrwec) : newInst;
		
		if (newInst == null)
			return false;
		
		//execute instructions & write the o/p to symbol table
		long t0 = System.nanoTime();
		executeInst(newInst, lrwec);
		long t1 = System.nanoTime();
		ec.setVariable(((ComputationCPInstruction)curr).output.getName(), lrwec.getVariable(LR_VAR));

		//put the result into the cache
		//Projected CT(Rewritten entry) = CT(last entry) + CT(rewrite), where CT = ComputeTime
		long totCT = _computeTime + (t1-t0);
		LineageCache.putMatrix(curr, ec, totCT);
		DMLScript.EXPLAIN = et; //TODO can't change this here
		
		//cleanup execution context
		lrwec.getVariables().removeAll();
		
		return true;
	}
	
	/*--------------------------------REWRITE METHODS------------------------------*/

	private static ArrayList<Instruction> rewriteTsmmCbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if(!isTsmmCbind(curr, ec, inCache))
			return null;
		
		// Create a transient read op over the cached tsmm result
		MatrixObject cachedEntry = toMatrixObject(inCache.get("lastMatrix"));
		lrwec.setVariable("cachedEntry", cachedEntry);
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		// Create rightIndex op to find the last matrix
		// TODO: For now assumption is that a single column is being appended in a loop
		//       Need to go down the lineage to find number of columns are being appended.
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		
		// Use X from cache, or create rightIndex
		Hop oldMatrix = inCache.containsKey("X") ?
			setupTReadCachedInput("X", inCache, lrwec) :
			HopRewriteUtils.createIndexingOp(newMatrix, 1L, mo.getNumRows(), 1L, mo.getNumColumns()-1);
		
		// Use deltaX from cache, or create rightIndex
		Hop lastCol = inCache.containsKey("deltaX") ?
			setupTReadCachedInput("deltaX", inCache, lrwec) :
			HopRewriteUtils.createIndexingOp(newMatrix, 1L, mo.getNumRows(), mo.getNumColumns(), mo.getNumColumns());
		
		Hop lrwHop = HopRewriteUtils.createPartialTsmmCbind(oldMatrix, lastCol, lastRes);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteTsmmCbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		
		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "X", "deltaX");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteTsmmCbindOnes (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// This is a specialization of rewriteTsmmCbind. This qualifies if 
		// the appended matrix is a column matrix of 1s (deltaX = 1s). 
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if(!isTsmmCbindOnes(curr, ec, inCache))
			return null;
		
		// Create a transient read op over the cached tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		// Create a transient read op over current input
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("newMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("newMatrix", mo);
		// rowTwo = colSums(newMatrix)
		AggUnaryOp rowTwo = HopRewriteUtils.createAggUnaryOp(newMatrix, AggOp.SUM, Direction.Col);
		// topRight = t(rowTwo[, 1:ncols-1])
		IndexingOp tmp = HopRewriteUtils.createIndexingOp(rowTwo, new LiteralOp(1), new LiteralOp(1), 
			new LiteralOp(1), new LiteralOp(mo.getNumColumns()-1));
		ReorgOp topRight = HopRewriteUtils.createTranspose(tmp);
		// rowOne = cbind(lastRes, topRight)
		BinaryOp rowOne = HopRewriteUtils.createBinary(lastRes, topRight, OpOp2.CBIND);
		// rbind(rowOne, rowTwo)
		BinaryOp lrwHop= HopRewriteUtils.createBinary(rowOne, rowTwo, OpOp2.RBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteTsmmCbindOnes APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}
	
	private static ArrayList<Instruction> rewriteTsmmRbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isTsmmRbind(curr, ec, inCache))
			return null;
		
		// Create a transient read op over the last tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		Hop lastRow;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastRow = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else
			lastRow = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(mo.getNumRows()), 
				new LiteralOp(mo.getNumRows()), new LiteralOp(1), new LiteralOp(mo.getNumColumns()));
		// tsmm(X + lastRow) = tsmm(X) + tsmm(lastRow)
		ReorgOp tlastRow = HopRewriteUtils.createTranspose(lastRow);
		AggBinaryOp tsmm_lr = HopRewriteUtils.createMatrixMultiply(tlastRow, lastRow);
		BinaryOp lrwHop = HopRewriteUtils.createBinary(lastRes, tsmm_lr, OpOp2.PLUS);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteTsmmRbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteTsmm2Cbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isTsmm2Cbind(curr, ec, inCache))
			return null;

		// Create a transient read op over the last tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		MatrixObject newmo = toMatrixObject(cachedEntry);
		lrwec.setVariable("cachedEntry", newmo);
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		
		// pull out the newly added column(2nd last) from the input matrix
		Hop lastCol;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastCol = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else
			lastCol = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), new LiteralOp(mo.getNumRows()), 
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
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteTsmm2Cbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX");
		
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteTsmm2CbindSameLeft (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		/* The difference between rewriteTsmm2Cbind and this rewrite is that the former applies
		 * when columns are increasingly appended where the later applies when different columns 
		 * are appended to a single base matrix.
		 */
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isTsmm2CbindSameLeft(curr, ec, inCache))
			return null;

		// Create a transient read op over the last tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		MatrixObject newmo = toMatrixObject(cachedEntry);
		lrwec.setVariable("cachedEntry", newmo);
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);

		// Create a transient read op over the input to this tsmm
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);

		// pull out the newly added column(2nd last) from the input matrix
		Hop lastCol;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastCol = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else
			lastCol = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), new LiteralOp(mo.getNumRows()), 
				new LiteralOp(mo.getNumColumns()-1), new LiteralOp(mo.getNumColumns()-1));

		// apply t(lastCol) on i/p matrix to get the result vectors.
		ReorgOp tlastCol = HopRewriteUtils.createTranspose(lastCol);
		AggBinaryOp newCol = HopRewriteUtils.createMatrixMultiply(tlastCol, newMatrix);
		ReorgOp tnewCol = HopRewriteUtils.createTranspose(newCol);

		// Replace the 2nd last row and column of the last tsmm resutl with the result vector.
		IndexingOp topLeft = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(1), new LiteralOp(newmo.getNumRows()-2), 
			new LiteralOp(1), new LiteralOp(newmo.getNumColumns()-2));
		IndexingOp topRight = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(1), new LiteralOp(newmo.getNumRows()-2), 
			new LiteralOp(newmo.getNumColumns()), new LiteralOp(newmo.getNumColumns()));
		IndexingOp bottomLeft = HopRewriteUtils.createIndexingOp(lastRes, new LiteralOp(newmo.getNumRows()), 
			new LiteralOp(newmo.getNumRows()), new LiteralOp(1), new LiteralOp(newmo.getNumColumns()-2));
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
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteTsmm2CbindSameLeft APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX");
		
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteMatMulRbindLeft (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isMatMulRbindLeft(curr, ec, inCache))
			return null;

		// Create a transient read op over the last ba+* result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("leftMatrix", moL);
		DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
		MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
		lrwec.setVariable("rightMatrix", moR);
		DataOp rightMatrix = HopRewriteUtils.createTransientRead("rightMatrix", moR);
		Hop lastRow;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastRow = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		lastRow = HopRewriteUtils.createIndexingOp(leftMatrix, new LiteralOp(moL.getNumRows()), 
			new LiteralOp(moL.getNumRows()), new LiteralOp(1), new LiteralOp(moL.getNumColumns()));
		// ba+*(X+lastRow, Y) = rbind(ba+*(X, Y), ba+*(lastRow, Y))
		AggBinaryOp rowTwo = HopRewriteUtils.createMatrixMultiply(lastRow, rightMatrix);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.RBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteMetMulRbindLeft APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteMatMulCbindRight (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isMatMulCbindRight(curr, ec, inCache))
			return null;

		// Create a transient read op over the last ba+* result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("leftMatrix", moL);
		DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
		MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
		lrwec.setVariable("rightMatrix", moR);
		DataOp rightMatrix = HopRewriteUtils.createTransientRead("rightMatrix", moR);
		Hop lastCol;
		// Use deltaY from cache, or create rightIndex
		if (inCache.containsKey("deltaY")) {
			MatrixBlock cachedRI = inCache.get("deltaY");
			lrwec.setVariable("deltaY", toMatrixObject(cachedRI));
			lastCol = HopRewriteUtils.createTransientRead("deltaY", cachedRI);
		}
		lastCol = HopRewriteUtils.createIndexingOp(rightMatrix, new LiteralOp(1), new LiteralOp(moR.getNumRows()), 
			new LiteralOp(moR.getNumColumns()), new LiteralOp(moR.getNumColumns()));
		// ba+*(X, cbind(Y, lastCol)) = cbind(ba+*(X, Y), ba+*(X, lastCol))
		AggBinaryOp colTwo = HopRewriteUtils.createMatrixMultiply(leftMatrix, lastCol);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, colTwo, OpOp2.CBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteMatMulCbindRight APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaY");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteMatMulCbindRightOnes (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// This is a specialization of rewriteMatMulCbindRight. This qualifies
		// if the right matrix is appended with a matrix of 1s (deltaY == 1s).
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isMatMulCbindRightOnes(curr, ec, inCache))
			return null;

		// Create a transient read op over the last ba+* result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("leftMatrix", moL);
		DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
		// ba+*(X, cbind(Y, ones)) = cbind(ba+*(X, Y), rowSums(X))
		AggUnaryOp colTwo = HopRewriteUtils.createAggUnaryOp(leftMatrix, AggOp.SUM, Direction.Row);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, colTwo, OpOp2.CBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteMatMulCbindRightOnes APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteElementMulRbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isElementMulRbind(curr, ec, inCache))
			return null;

		// Create a transient read op over the last * result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		Hop lastRowL, lastRowR;
		// Use deltaX and deltaY from cache, or create rightIndices
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastRowL = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else {
			MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
			lrwec.setVariable("leftMatrix", moL);
			DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
			lastRowL = HopRewriteUtils.createIndexingOp(leftMatrix, new LiteralOp(moL.getNumRows()), 
				new LiteralOp(moL.getNumRows()), new LiteralOp(1), new LiteralOp(moL.getNumColumns()));
		}
		if (inCache.containsKey("deltaY")) {
			MatrixBlock cachedRI = inCache.get("deltaY");
			lrwec.setVariable("deltaY", toMatrixObject(cachedRI));
			lastRowR = HopRewriteUtils.createTransientRead("deltaY", cachedRI);
		}
		else {
			MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
			lrwec.setVariable("rightMatrix", moR);
			DataOp rightMatrix = HopRewriteUtils.createTransientRead("rightMatrix", moR);
			lastRowR = HopRewriteUtils.createIndexingOp(rightMatrix, new LiteralOp(moR.getNumRows()), 
				new LiteralOp(moR.getNumRows()), new LiteralOp(1), new LiteralOp(moR.getNumColumns()));
		}
		// *(X+lastRowL, Y+lastRowR) = rbind(*(X, Y), *(lastRowL, lastRowR))
		BinaryOp rowTwo = HopRewriteUtils.createBinary(lastRowL, lastRowR, OpOp2.MULT);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.RBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteElementMulRbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX", "deltaY");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteElementMulCbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isElementMulCbind(curr, ec, inCache))
			return null;

		// Create a transient read op over the last * result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		Hop lastColL, lastColR;
		// Use deltaX and deltaY from cache, or create rightIndices
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastColL = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else {
			MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
			lrwec.setVariable("leftMatrix", moL);
			DataOp leftMatrix = HopRewriteUtils.createTransientRead("leftMatrix", moL);
			lastColL = HopRewriteUtils.createIndexingOp(leftMatrix, new LiteralOp(1), 
				new LiteralOp(moL.getNumRows()), new LiteralOp(moL.getNumColumns()), new LiteralOp(moL.getNumColumns()));
		}
		if (inCache.containsKey("deltaY")) {
			MatrixBlock cachedRI = inCache.get("deltaY");
			lrwec.setVariable("deltaY", toMatrixObject(cachedRI));
			lastColR = HopRewriteUtils.createTransientRead("deltaY", cachedRI);
		}
		else {
			MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
			lrwec.setVariable("rightMatrix", moR);
			DataOp rightMatrix = HopRewriteUtils.createTransientRead("rightMatrix", moR);
			lastColR = HopRewriteUtils.createIndexingOp(rightMatrix, new LiteralOp(1), 
				new LiteralOp(moR.getNumRows()), new LiteralOp(moR.getNumColumns()), new LiteralOp(moR.getNumColumns()));
		}
		// *(X+lastRowL, Y+lastRowR) = cbind(*(X, Y), *(lastColL, lastColR))
		BinaryOp rowTwo = HopRewriteUtils.createBinary(lastColL, lastColR, OpOp2.MULT);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.CBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteElementMulCbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX", "deltaY");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}
	
	private static ArrayList<Instruction> rewriteAggregateCbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isAggCbind (curr, ec, inCache))
			return null;
		
		// Create a transient read op over the last * result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", toMatrixObject(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		HashMap<String, String> params = ((ParameterizedBuiltinCPInstruction)curr).getParameterMap();
		MatrixObject mo = ec.getMatrixObject(params.get(Statement.GAGG_TARGET));
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		MatrixObject moG = ec.getMatrixObject(params.get(Statement.GAGG_GROUPS));
		lrwec.setVariable("groups", moG);
		DataOp groups = HopRewriteUtils.createTransientRead("groups", moG);
		String fn = params.get(Statement.GAGG_FN);
		int ngroups = (params.get(Statement.GAGG_NUM_GROUPS) != null) ? 
			(int)Double.parseDouble(params.get(Statement.GAGG_NUM_GROUPS)) : -1;
		Hop lastCol;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", toMatrixObject(cachedRI));
			lastCol = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else
			lastCol = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), new LiteralOp(mo.getNumRows()), 
				new LiteralOp(mo.getNumColumns()), new LiteralOp(mo.getNumColumns()));
		// aggregate(target=X+lastCol,...) = cbind(aggregate(target=X,...), aggregate(target=lastCol,...))
		LinkedHashMap<String, Hop> args = new LinkedHashMap<>();
		args.put("target", lastCol);
		args.put("groups", groups);
		args.put("fn", new LiteralOp(fn));
		if (ngroups != -1) 
			args.put("ngroups", new LiteralOp(ngroups));
		ParameterizedBuiltinOp rowTwo = HopRewriteUtils.createParameterizedBuiltinOp(newMatrix, args, ParamBuiltinOp.GROUPEDAGG);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.CBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteElementMulCbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "deltaX");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewriteIndexingMatMul (Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		/* This rewrite replaces the indexed matrix with its source as an 
		 * input to matrix multiplication, with the hope that in future 
		 * iterations all the outputs can be sliced out from the cached 
		 * result (e.g. PCA in a loop)
		 * Note: this particular rewrite needs to cache the compensation plan
		 * execution results (unlike other rewrites) to be effective.
		 * TODO: Generalize for all cases and move to compiler
		 */
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isIndexingMatMul (curr, ec, inCache))
			return null;
		
		// Create a transient read op over the input to rightIndex
		MatrixBlock indexSource = inCache.get("indexSource");
		lrwec.setVariable("indexSource", toMatrixObject(indexSource));
		DataOp input2Index = HopRewriteUtils.createTransientRead("indexSource", indexSource);
		// Create or read the matrix multiplication
		Hop matMultRes;
		MatrixObject moL = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		if (inCache.containsKey("BigMatMult")) {
			MatrixObject BigMatMult = toMatrixObject(inCache.get("BigMatMult"));
			lrwec.setVariable("BigMatMult", BigMatMult);
			matMultRes = HopRewriteUtils.createTransientRead("BigMatMult", BigMatMult);
		}
		else {
			lrwec.setVariable("left", moL);
			DataOp leftMatrix = HopRewriteUtils.createTransientRead("left", moL);
			matMultRes = HopRewriteUtils.createMatrixMultiply(leftMatrix, input2Index);
			// Perform the multiplication once and cache for future iterations.
		}
		// Gather the indexing parameters.
		MatrixObject moR = ec.getMatrixObject(((ComputationCPInstruction)curr).input2);
		IndexingOp lrwHop = HopRewriteUtils.createIndexingOp(matMultRes, new LiteralOp(1), 
				new LiteralOp(moL.getNumRows()), new LiteralOp(1), new LiteralOp(moR.getNumColumns()));
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewriteIndexingMatMul APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		// Keep reuse enabled
		_disableReuse = false;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "indexSource", "BigMatMult");
		
		if (DMLScript.STATISTICS)
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}

	private static ArrayList<Instruction> rewritePcaTsmm(Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isPcaTsmm(curr, ec, inCache))
			return null;

		// Create a transient read op over the last tsmm result
		MatrixObject newmo = toMatrixObject(inCache.get("lastMatrix"));
		lrwec.setVariable("cachedEntry", newmo);
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", newmo);

		// Create a transient read op over this tsmm's input 
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		
		// Index out the added column from the projected matrix
		MatrixObject projmo = toMatrixObject(inCache.get("projected"));
		lrwec.setVariable("projected", projmo);
		DataOp projRes = HopRewriteUtils.createTransientRead("projected", projmo);
		IndexingOp lastCol = HopRewriteUtils.createIndexingOp(projRes, new LiteralOp(1), new LiteralOp(projmo.getNumRows()), 
				new LiteralOp(projmo.getNumColumns()), new LiteralOp(projmo.getNumColumns()));
		
		// Apply t(lastCol) on i/p matrix to get the result vectors.
		ReorgOp tlastCol = HopRewriteUtils.createTranspose(lastCol);
		AggBinaryOp newCol = HopRewriteUtils.createMatrixMultiply(tlastCol, newMatrix);
		ReorgOp tnewCol = HopRewriteUtils.createTranspose(newCol);
		
		// Push the result row & column inside the cashed block as 2nd last row and col respectively.
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
		
		// Generate runtime instructions
		if (LOG.isDebugEnabled())
			LOG.debug("LINEAGE REWRITE rewritePcaTsmm APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);
		_disableReuse = true;

		// cleanup buffer pool
		addRmvarInstructions(inst, lrwec, "cachedEntry", "projected");
		
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementPRewrites();
		return inst;
	}
	
	/*------------------------REWRITE APPLICABILITY CHECKS-------------------------*/

	private static boolean isTsmmCbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec)) {
			return false;
		}

		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.TSMM.toString())) {
			LineageItem source = item.getInputs()[0];
			if (source.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				// create tsmm lineage on top of the input of last append
				LineageItem input1 = source.getInputs()[0];
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {input1});
				if (LineageCache.probe(tmp)) { 
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the old matrix in cache
				if( LineageCache.probe(input1) )
					inCache.put("X", LineageCache.getMatrix(input1));
				// look for the appended column in cache
				if (LineageCache.probe(source.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(source.getInputs()[1]));
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isTsmmCbindOnes(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec)) {
			return false;
		}

		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.TSMM.toString())) {
			LineageItem source = item.getInputs()[0];
			if (source.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				// check if the appended column is a matrix of 1s
				LineageItem input2 = source.getInputs()[1];
				if (input2.getType() != LineageItemType.Creation)
					return false;
				Instruction ins = InstructionParser.parseSingleInstruction(input2.getData());
				if (!((DataGenCPInstruction)ins).isOnesCol())
					return false;
				// create tsmm lineage on top of the input of last append
				LineageItem input1 = source.getInputs()[0];
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {input1});
				if( LineageCache.probe(input1) )
					inCache.put("X", LineageCache.getMatrix(input1));
				if (LineageCache.probe(tmp)) { 
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime += LineageCache.getEntry(tmp)._computeTime;
				}
			}
		}
		// return true only if the last tsmm result is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isTsmmRbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the input to tsmm came from rbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.TSMM.toString())) {
			LineageItem source = item.getInputs()[0];
			if (source.getOpcode().equalsIgnoreCase(Opcodes.RBIND.toString())) {
				// create tsmm lineage on top of the input of last append
				LineageItem input1 = source.getInputs()[0];
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {input1});
				if (LineageCache.probe(tmp)) { 
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the appended column in cache
				if (source.getInputs().length>1 && LineageCache.probe(source.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(source.getInputs()[1]));
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}
	
	private static boolean isTsmm2Cbind (Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		//TODO: support nary cbind
		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		// look for two consecutive cbinds
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.TSMM.toString())) {
			LineageItem source = item.getInputs()[0];
			if (source.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				LineageItem input = source.getInputs()[0];
				if (input.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
					LineageItem L2appin1 = input.getInputs()[0]; 
					LineageItem tmp = new LineageItem(Opcodes.CBIND.toString(), new LineageItem[] {L2appin1, source.getInputs()[1]});
					LineageItem toProbe = new LineageItem(curr.getOpcode(), new LineageItem[] {tmp});
					if (LineageCache.probe(toProbe)) { 
						inCache.put("lastMatrix", LineageCache.getMatrix(toProbe));
						_computeTime = LineageCache.getEntry(toProbe)._computeTime;
					}
					// look for the appended column in cache
					if (LineageCache.probe(input.getInputs()[1])) 
						inCache.put("deltaX", LineageCache.getMatrix(input.getInputs()[1]));
				}
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isTsmm2CbindSameLeft (Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		//TODO: support nary cbind
		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		// look for two consecutive cbinds
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.TSMM.toString())) {
			LineageItem source = item.getInputs()[0];
			if (source.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				LineageItem input = source.getInputs()[0];
				if (input.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
					LineageItem L2appin1 = input.getInputs()[0]; 
					if (!L2appin1.getOpcode().equalsIgnoreCase("rightIndex"))
						return false;
					LineageItem RI = input.getInputs()[1];
					if (LineageCache.probe(RI))
						inCache.put("deltaX", LineageCache.getMatrix(RI));
					LineageItem cu = RI.getInputs()[4];
					LineageItem old_cu = reduceColByOne(cu);
					LineageItem old_RI = new LineageItem("rightIndex", new LineageItem[] {RI.getInputs()[0],
							RI.getInputs()[1], RI.getInputs()[2], old_cu, old_cu});
					LineageItem old_cbind = new LineageItem(Opcodes.CBIND.toString(), new LineageItem[] {L2appin1, old_RI});
					LineageItem tmp = new LineageItem(Opcodes.CBIND.toString(), new LineageItem[] {old_cbind, source.getInputs()[1]});
					LineageItem toProbe = new LineageItem(curr.getOpcode(), new LineageItem[] {tmp});
					if (LineageCache.probe(toProbe)) { 
						inCache.put("lastMatrix", LineageCache.getMatrix(toProbe));
						_computeTime = LineageCache.getEntry(toProbe)._computeTime;
					}
				}
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isMatMulRbindLeft(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the left input to ba+* came from rbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.MMULT.toString())) {
			LineageItem left= item.getInputs()[0];
			LineageItem right = item.getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase(Opcodes.RBIND.toString())){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of rbind = X
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {leftSource, right});
				if (LineageCache.probe(tmp)) {
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the appended column in cache
				if (LineageCache.probe(left.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(left.getInputs()[1]));
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isMatMulCbindRight(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the right input to ba+* came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.MMULT.toString())) {
			LineageItem left = item.getInputs()[0];
			LineageItem right = item.getInputs()[1];
			if (right.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				LineageItem rightSource = right.getInputs()[0]; //left inpur of rbind = X
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {left, rightSource});
				if (LineageCache.probe(tmp)) {
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the appended column in cache
				if (LineageCache.probe(right.getInputs()[1])) 
					inCache.put("deltaY", LineageCache.getMatrix(right.getInputs()[1]));
			}
		}
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isMatMulCbindRightOnes(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the right input to ba+* came from cbind of a matrix and ones.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.MMULT.toString())) {
			LineageItem left = item.getInputs()[0];
			LineageItem right = item.getInputs()[1];
			if (right.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				LineageItem rightSource1 = right.getInputs()[0]; //left input of cbind is X
				LineageItem rightSource2 = right.getInputs()[1]; 
				// check if the right input to cbind is a matrix of 1s.
				if (rightSource2.getType() != LineageItemType.Creation)
					return false;
				Instruction ins = InstructionParser.parseSingleInstruction(rightSource2.getData());
				if (!((DataGenCPInstruction)ins).isOnesCol())
					return false;
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {left, rightSource1});
				if (LineageCache.probe(tmp)) {
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
			}
		}
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isElementMulRbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the inputs to * came from rbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.MULT.toString())) {
			LineageItem left= item.getInputs()[0];
			LineageItem right = item.getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase(Opcodes.RBIND.toString()) && right.getOpcode().equalsIgnoreCase(Opcodes.RBIND.toString())){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of rbind = X
				LineageItem rightSource = right.getInputs()[0]; //right inpur of rbind = Y 
				// create * lineage on top of the input of last append
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {leftSource, rightSource});
				if (LineageCache.probe(tmp)) {
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the appended rows in cache
				if (LineageCache.probe(left.getInputs()[1]))
					inCache.put("deltaX", LineageCache.getMatrix(left.getInputs()[1]));
				if (LineageCache.probe(right.getInputs()[1]))
					inCache.put("deltaY", LineageCache.getMatrix(right.getInputs()[1]));
			}
		}
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isElementMulCbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the inputs to * came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.MULT.toString())) {
			LineageItem left= item.getInputs()[0];
			LineageItem right = item.getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString()) && right.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of cbind = X
				LineageItem rightSource = right.getInputs()[0]; //right inpur of cbind = Y 
				// create * lineage on top of the input of last append
				LineageItem tmp = new LineageItem(curr.getOpcode(), new LineageItem[] {leftSource, rightSource});
				if (LineageCache.probe(tmp)) {
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the appended columns in cache
				if (LineageCache.probe(left.getInputs()[1]))
					inCache.put("deltaX", LineageCache.getMatrix(left.getInputs()[1]));
				if (LineageCache.probe(right.getInputs()[1]))
					inCache.put("deltaY", LineageCache.getMatrix(right.getInputs()[1]));
			}
		}
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isAggCbind (Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec)) {
			return false;
		}

		// If the input to groupedagg came from cbind, look for both the inputs in cache.
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.GROUPEDAGG.toString())) {
			LineageItem target = item.getInputs()[0];
			LineageItem groups = item.getInputs()[1];
			LineageItem weights = item.getInputs()[2];
			LineageItem fn = item.getInputs()[3];
			LineageItem ngroups = item.getInputs()[4];
			if (target.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				// create groupedagg lineage on top of the input of last append
				LineageItem input1 = target.getInputs()[0];
				LineageItem tmp = new LineageItem(curr.getOpcode(), 
						new LineageItem[] {input1, groups, weights, fn, ngroups});
				if (LineageCache.probe(tmp)) {
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
					_computeTime = LineageCache.getEntry(tmp)._computeTime;
				}
				// look for the appended column in cache
				if (LineageCache.probe(target.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(target.getInputs()[1]));
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}
	
	private static boolean isIndexingMatMul(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache) {
		if (!LineageCacheConfig.isReusable(curr, ec)) {
			return false;
		}
		/* rightIndex -> ba+* is to generic. 
		 * Use ba+* -> rightIndex -> ba+* to avoid false positives.
		 * TODO: generalized but robust applicability function
		 */

		// Check if the right input of ba+* came from rightindex
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.MMULT.toString())) {
			LineageItem left = item.getInputs()[0];
			LineageItem right = item.getInputs()[1];
			if (right.getOpcode().equalsIgnoreCase("rightIndex")) {
				LineageItem indexSource = right.getInputs()[0];
				if (LineageCache.probe(indexSource) && indexSource.getOpcode().equalsIgnoreCase(Opcodes.MMULT.toString())) {
					inCache.put("indexSource", LineageCache.getMatrix(indexSource));
					_computeTime = LineageCache.getEntry(indexSource)._computeTime;
				}
				LineageItem tmp = new LineageItem(item.getOpcode(), new LineageItem[] {left, indexSource});
				if (LineageCache.probe(tmp))
					inCache.put("BigMatMult", LineageCache.getMatrix(tmp));
			}
		}
		// return true only if the input to rightIndex is found
		return inCache.containsKey("indexSource") ? true : false;
	}

	private static boolean isPcaTsmm(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache) {
		if (!LineageCacheConfig.isReusable(curr, ec)) {
			return false;
		}
		
		LineageItem item = ((ComputationCPInstruction) curr).getLineageItem(ec).getValue();
		if (curr.getOpcode().equalsIgnoreCase(Opcodes.TSMM.toString())) {
			LineageItem src1 = item.getInputs()[0];
			if (src1.getOpcode().equalsIgnoreCase(Opcodes.CBIND.toString())) {
				LineageItem src21 = src1.getInputs()[0];
				LineageItem src22 = src1.getInputs()[1]; //ones
				if (src21.getOpcode().equalsIgnoreCase(Opcodes.MMULT.toString())) {
					if (LineageCache.probe(src21)) {
						inCache.put("projected", LineageCache.getMatrix(src21));
						_computeTime = LineageCache.getEntry(src21)._computeTime;
					}
				
					LineageItem src31 = src21.getInputs()[1];
					LineageItem src32 = src21.getInputs()[0];
					if (src31.getOpcode().equalsIgnoreCase("rightIndex")) {
						LineageItem cu = src31.getInputs()[4];
						//TODO: delta with more than one column
						LineageItem old_cu = reduceColByOne(cu);
						LineageItem old_RI = new LineageItem("rightIndex", new LineageItem[] {src31.getInputs()[0], 
								src31.getInputs()[1], src31.getInputs()[2], src31.getInputs()[3], old_cu});
						LineageItem old_ba = new LineageItem(Opcodes.MMULT.toString(), new LineageItem[] {src32, old_RI});
						LineageItem old_cbind = new LineageItem(Opcodes.CBIND.toString(), new LineageItem[] {old_ba, src22});
						LineageItem old_tsmm = new LineageItem(Opcodes.TSMM.toString(), new LineageItem[] {old_cbind});
						if (LineageCache.probe(old_tsmm)) {
							inCache.put("lastMatrix", LineageCache.getMatrix(old_tsmm));
							_computeTime += LineageCache.getEntry(old_tsmm)._computeTime;
						}
					}
				}
			}
		}
		return inCache.containsKey("projected") && inCache.containsKey("lastMatrix");
	}

	/*----------------------INSTRUCTIONS GENERATION & EXECUTION-----------------------*/

	private static ArrayList<Instruction> genInst(Hop hops, ExecutionContext ec) {
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(hops, ec.getVariables(), null, true, true, 0);
		if (LOG.isDebugEnabled()) {
			LOG.debug("COMPENSATION PLAN: ");
			LOG.debug("EXPLAIN LINEAGE REWRITE (HOP) \n" + Explain.explain(hops,1));
			LOG.debug("EXPLAIN LINEAGE REWRITE (INSTRUCTION) \n" + Explain.explain(newInst,1));
		}
		return newInst;
	}
	
	private static DataOp setupTReadCachedInput(String name, Map<String, MatrixBlock> inCache, ExecutionContext ec) {
		MatrixBlock cachedRI = inCache.get(name);
		ec.setVariable(name, toMatrixObject(cachedRI));
		return HopRewriteUtils.createTransientRead(name, cachedRI);
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
			if (_disableReuse)
				LineageCacheConfig.shutdownReuse();
			pb.execute(lrwec);
			if (_disableReuse)
				LineageCacheConfig.restartReuse(oldReuseOption);
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Error executing lineage rewrites" , e);
		}
	}

	/*-------------------------------UTILITY METHODS----------------------------------*/
	
	private static MatrixObject toMatrixObject(MatrixBlock mb) {
		MetaData md = new MetaData(mb.getDataCharacteristics());
		MatrixObject mo = new MatrixObject(ValueType.FP64, null, md);
		mo.acquireModify(mb);
		mo.release();
		return mo;
	}
	
	private static void addRmvarInstructions(ArrayList<Instruction> inst, ExecutionContext ec, String... varnames) {
		//note: we can't directly call rmvar because the instructions are not executed yet
		ArrayList<String> tmp = new ArrayList<>();
		for(String varname : varnames)
			if(ec.containsVariable(varname))
				tmp.add(varname);
		inst.add(VariableCPInstruction.prepareRemoveInstruction(tmp.toArray(new String[0])));
	}
	
	private static LineageItem reduceColByOne(LineageItem cu) {
		String old_data = null;
		try {
			String data = cu.getData();  //xx·SCALAR·INT64·true
			String[] parts = data.split(Instruction.VALUETYPE_PREFIX);
			float cuNum = Float.valueOf(parts[0]);
			parts[0] = String.valueOf((int)cuNum-1);
			old_data = InstructionUtils.concatOperandParts(parts);
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Error reading 'cu' from RightIndex instruction" , e);
		}
		return(new LineageItem(old_data));
	}

	private static ExecutionContext getExecutionContext() {
		if( _lrEC == null )
			_lrEC = ExecutionContextFactory.createContext();
		return _lrEC;
	}

	private static BasicProgramBlock getProgramBlock() {
		if( _lrPB == null )
			_lrPB = new BasicProgramBlock(new Program());
		return _lrPB;
	}
}
