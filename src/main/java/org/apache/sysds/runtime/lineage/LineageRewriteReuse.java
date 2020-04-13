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
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.AggBinaryOp;
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
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MetaData;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Explain.ExplainType;

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
		//tsmm(cbind(X, deltaX)) -> rbind(cbind(tsmm(X), t(X) %*% deltaX), cbind(t(deltaX) %*%X, tsmm(deltaX)))
		ArrayList<Instruction> newInst = rewriteTsmmCbind(curr, ec, lrwec);
		//tsmm(cbind(cbind(X, deltaX), ones)) -> TODO
		newInst = (newInst == null) ? rewriteTsmm2Cbind(curr, ec, lrwec) : newInst;
		//tsmm(rbind(X, deltaX)) -> tsmm(X) + tsmm(deltaX)
		newInst = (newInst == null) ? rewriteTsmmRbind(curr, ec, lrwec) : newInst;
		//rbind(X,deltaX) %*% Y -> rbind(X %*% Y, deltaX %*% Y)
		newInst = (newInst == null) ? rewriteMatMulRbindLeft(curr, ec, lrwec) : newInst;
		//X %*% cbind(Y,deltaY)) -> cbind(X %*% Y, X %*% deltaY)
		newInst = (newInst == null) ? rewriteMatMulCbindRight(curr, ec, lrwec) : newInst;
		//rbind(X, deltaX) * rbind(Y, deltaY) -> rbind(X * Y, deltaX * deltaY)
		newInst = (newInst == null) ? rewriteElementMulRbind(curr, ec, lrwec) : newInst;
		//cbind(X, deltaX) * cbind(Y, deltaY) -> cbind(X * Y, deltaX * deltaY)
		newInst = (newInst == null) ? rewriteElementMulCbind(curr, ec, lrwec) : newInst;
		//aggregate(target=cbind(X, deltaX,...) = cbind(aggregate(target=X,...), aggregate(target=deltaX,...)) for same agg function
		newInst = (newInst == null) ? rewriteAggregateCbind(curr, ec, lrwec) : newInst;
		
		if (newInst == null)
			return false;
		
		//execute instructions & write the o/p to symbol table
		executeInst(newInst, lrwec);
		ec.setVariable(((ComputationCPInstruction)curr).output.getName(), lrwec.getVariable(LR_VAR));

		//put the result into the cache
		LineageCache.putMatrix(curr, ec);
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
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the cached tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		// Create rightIndex op to find the last matrix
		// TODO: For now assumption is that a single column is being appended in a loop
		//       Need to go down the lineage to find number of columns are being appended.
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		IndexingOp oldMatrix = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), 
				new LiteralOp(mo.getNumRows()), new LiteralOp(1), new LiteralOp(mo.getNumColumns()-1));
		Hop lastCol;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
			lastCol = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		else
			lastCol = HopRewriteUtils.createIndexingOp(newMatrix, new LiteralOp(1), new LiteralOp(mo.getNumRows()), 
					new LiteralOp(mo.getNumColumns()), new LiteralOp(mo.getNumColumns()));
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
		LOG.debug("LINEAGE REWRITE rewriteTsmmCbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}
	
	private static ArrayList<Instruction> rewriteTsmmRbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isTsmmRbind(curr, ec, inCache))
			return null;
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		MatrixObject mo = ec.getMatrixObject(((ComputationCPInstruction)curr).input1);
		lrwec.setVariable("oldMatrix", mo);
		DataOp newMatrix = HopRewriteUtils.createTransientRead("oldMatrix", mo);
		Hop lastRow;
		// Use deltaX from cache, or create rightIndex
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
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
		LOG.debug("LINEAGE REWRITE rewriteTsmmRbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}

	private static ArrayList<Instruction> rewriteTsmm2Cbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isTsmm2Cbind(curr, ec, inCache))
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last tsmm result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		MatrixObject newmo = convMBtoMO(cachedEntry);
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
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
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
		LOG.debug("LINEAGE REWRITE rewriteTsmm2Cbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}

	private static ArrayList<Instruction> rewriteMatMulRbindLeft (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isMatMulRbindLeft(curr, ec, inCache))
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last ba+* result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
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
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
			lastRow = HopRewriteUtils.createTransientRead("deltaX", cachedRI);
		}
		lastRow = HopRewriteUtils.createIndexingOp(leftMatrix, new LiteralOp(moL.getNumRows()), 
				new LiteralOp(moL.getNumRows()), new LiteralOp(1), new LiteralOp(moL.getNumColumns()));
		// ba+*(X+lastRow, Y) = rbind(ba+*(X, Y), ba+*(lastRow, Y))
		AggBinaryOp rowTwo = HopRewriteUtils.createMatrixMultiply(lastRow, rightMatrix);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.RBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteMetMulRbindLeft APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}

	private static ArrayList<Instruction> rewriteMatMulCbindRight (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isMatMulCbindRight(curr, ec, inCache))
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last ba+* result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
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
			lrwec.setVariable("deltaY", convMBtoMO(cachedRI));
			lastCol = HopRewriteUtils.createTransientRead("deltaY", cachedRI);
		}
		lastCol = HopRewriteUtils.createIndexingOp(rightMatrix, new LiteralOp(1), new LiteralOp(moR.getNumRows()), 
				new LiteralOp(moR.getNumColumns()), new LiteralOp(moR.getNumColumns()));
		// ba+*(X, Y+lastCol) = cbind(ba+*(X, Y), ba+*(X, lastCol))
		AggBinaryOp rowTwo = HopRewriteUtils.createMatrixMultiply(leftMatrix, lastCol);
		BinaryOp lrwHop= HopRewriteUtils.createBinary(lastRes, rowTwo, OpOp2.CBIND);
		DataOp lrwWrite = HopRewriteUtils.createTransientWrite(LR_VAR, lrwHop);

		// generate runtime instructions
		LOG.debug("LINEAGE REWRITE rewriteMatMulCbindRight APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}

	private static ArrayList<Instruction> rewriteElementMulRbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isElementMulRbind(curr, ec, inCache))
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last * result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		Hop lastRowL, lastRowR;
		// Use deltaX and deltaY from cache, or create rightIndices
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
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
			lrwec.setVariable("deltaY", convMBtoMO(cachedRI));
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
		LOG.debug("LINEAGE REWRITE rewriteElementMulRbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}

	private static ArrayList<Instruction> rewriteElementMulCbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec) 
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isElementMulCbind(curr, ec, inCache))
			return null;

		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last * result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
		DataOp lastRes = HopRewriteUtils.createTransientRead("cachedEntry", cachedEntry);
		//TODO: support for block of rows
		Hop lastColL, lastColR;
		// Use deltaX and deltaY from cache, or create rightIndices
		if (inCache.containsKey("deltaX")) {
			MatrixBlock cachedRI = inCache.get("deltaX");
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
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
			lrwec.setVariable("deltaY", convMBtoMO(cachedRI));
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
		LOG.debug("LINEAGE REWRITE rewriteElementMulCbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}
	
	private static ArrayList<Instruction> rewriteAggregateCbind (Instruction curr, ExecutionContext ec, ExecutionContext lrwec)
	{
		// Check the applicability of this rewrite.
		Map<String, MatrixBlock> inCache = new HashMap<>();
		if (!isAggCbind (curr, ec, inCache))
			return null;
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		// Create a transient read op over the last * result
		MatrixBlock cachedEntry = inCache.get("lastMatrix");
		lrwec.setVariable("cachedEntry", convMBtoMO(cachedEntry));
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
			lrwec.setVariable("deltaX", convMBtoMO(cachedRI));
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
		LOG.debug("LINEAGE REWRITE rewriteElementMulCbind APPLIED");
		ArrayList<Instruction> inst = genInst(lrwWrite, lrwec);

		if (DMLScript.STATISTICS) {
			LineageCacheStatistics.incrementPRewriteTime(System.nanoTime() - t0);
			LineageCacheStatistics.incrementPRewrites();
		}
		return inst;
	}
	
	/*------------------------REWRITE APPLICABILITY CHECKS-------------------------*/

	private static boolean isTsmmCbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec)) {
			return false;
		}

		// If the input to tsmm came from cbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		LineageItem item = items[0];
		for (LineageItem source : item.getInputs())
			if (source.getOpcode().equalsIgnoreCase("cbind")) {
				//for (LineageItem input : source.getInputs()) {
				// create tsmm lineage on top of the input of last append
				LineageItem input1 = source.getInputs()[0];
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {input1});
				if (LineageCache.probe(tmp)) 
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
				// look for the appended column in cache
				if (LineageCache.probe(source.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(source.getInputs()[1]));
			}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isTsmmRbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the input to tsmm came from rbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		LineageItem item = items[0];
		for (LineageItem source : item.getInputs())
			if (source.getOpcode().equalsIgnoreCase("rbind")) {
				// create tsmm lineage on top of the input of last append
				LineageItem input1 = source.getInputs()[0];
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {input1});
				if (LineageCache.probe(tmp)) 
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
				// look for the appended column in cache
				if (LineageCache.probe(source.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(source.getInputs()[1]));
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
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		LineageItem item = items[0];
		// look for two consecutive cbinds
		for (LineageItem source : item.getInputs())
			if (source.getOpcode().equalsIgnoreCase("cbind")) {
				LineageItem input = source.getInputs()[0];
				if (input.getOpcode().equalsIgnoreCase("cbind")) {
					LineageItem L2appin1 = input.getInputs()[0]; 
					LineageItem tmp = new LineageItem("comb", "cbind", new LineageItem[] {L2appin1, source.getInputs()[1]});
					LineageItem toProbe = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {tmp});
					if (LineageCache.probe(toProbe)) 
						inCache.put("lastMatrix", LineageCache.getMatrix(toProbe));
					// look for the appended column in cache
					if (LineageCache.probe(input.getInputs()[1])) 
						inCache.put("deltaX", LineageCache.getMatrix(input.getInputs()[1]));
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
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("ba+*")) {
			LineageItem left= items[0].getInputs()[0];
			LineageItem right = items[0].getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase("rbind")){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of rbind = X
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {leftSource, right});
				if (LineageCache.probe(tmp))
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
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
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("ba+*")) {
			LineageItem left = items[0].getInputs()[0];
			LineageItem right = items[0].getInputs()[1];
			if (right.getOpcode().equalsIgnoreCase("cbind")) {
				LineageItem rightSource = right.getInputs()[0]; //left inpur of rbind = X
				// create ba+* lineage on top of the input of last append
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {left, rightSource});
				if (LineageCache.probe(tmp))
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
				// look for the appended column in cache
				if (LineageCache.probe(right.getInputs()[1])) 
					inCache.put("deltaY", LineageCache.getMatrix(right.getInputs()[1]));
			}
		}
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	private static boolean isElementMulRbind(Instruction curr, ExecutionContext ec, Map<String, MatrixBlock> inCache)
	{
		if (!LineageCacheConfig.isReusable(curr, ec))
			return false;

		// If the inputs to * came from rbind, look for both the inputs in cache.
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("*")) {
			LineageItem left= items[0].getInputs()[0];
			LineageItem right = items[0].getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase("rbind") && right.getOpcode().equalsIgnoreCase("rbind")){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of rbind = X
				LineageItem rightSource = right.getInputs()[0]; //right inpur of rbind = Y 
				// create * lineage on top of the input of last append
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {leftSource, rightSource});
				if (LineageCache.probe(tmp))
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
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
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("*")) {
			LineageItem left= items[0].getInputs()[0];
			LineageItem right = items[0].getInputs()[1];
			if (left.getOpcode().equalsIgnoreCase("cbind") && right.getOpcode().equalsIgnoreCase("cbind")){
				LineageItem leftSource = left.getInputs()[0]; //left inpur of cbind = X
				LineageItem rightSource = right.getInputs()[0]; //right inpur of cbind = Y 
				// create * lineage on top of the input of last append
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), new LineageItem[] {leftSource, rightSource});
				if (LineageCache.probe(tmp))
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
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
		LineageItem[] items = ((ComputationCPInstruction) curr).getLineageItems(ec);
		if (curr.getOpcode().equalsIgnoreCase("groupedagg")) {
			LineageItem target = items[0].getInputs()[0];
			LineageItem groups = items[0].getInputs()[1];
			LineageItem weights = items[0].getInputs()[2];
			LineageItem fn = items[0].getInputs()[3];
			LineageItem ngroups = items[0].getInputs()[4];
			if (target.getOpcode().equalsIgnoreCase("cbind")) {
				// create groupedagg lineage on top of the input of last append
				LineageItem input1 = target.getInputs()[0];
				LineageItem tmp = new LineageItem("toProbe", curr.getOpcode(), 
						new LineageItem[] {input1, groups, weights, fn, ngroups});
				if (LineageCache.probe(tmp)) 
					inCache.put("lastMatrix", LineageCache.getMatrix(tmp));
				// look for the appended column in cache
				if (LineageCache.probe(target.getInputs()[1])) 
					inCache.put("deltaX", LineageCache.getMatrix(target.getInputs()[1]));
			}
		}
		// return true only if the last tsmm is found
		return inCache.containsKey("lastMatrix") ? true : false;
	}

	/*----------------------INSTRUCTIONS GENERATION & EXECUTION-----------------------*/

	private static ArrayList<Instruction> genInst(Hop hops, ExecutionContext ec) {
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(hops, ec.getVariables(), null, true, true, 0);
		if (DMLScript.STATISTICS) 
			LineageCacheStatistics.incrementPRwExecTime(System.nanoTime()-t0);
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
			long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
			//execute instructions
			BasicProgramBlock pb = getProgramBlock();
			pb.setInstructions(newInst);
			ReuseCacheType oldReuseOption = DMLScript.LINEAGE_REUSE;
			LineageCacheConfig.shutdownReuse();
			pb.execute(lrwec);
			LineageCacheConfig.restartReuse(oldReuseOption);
			if (DMLScript.STATISTICS) 
				LineageCacheStatistics.incrementPRwExecTime(System.nanoTime()-t0);
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
			_lrPB = new BasicProgramBlock(new Program());
		return _lrPB;
	}
}