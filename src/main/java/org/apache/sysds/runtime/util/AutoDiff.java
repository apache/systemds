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

package org.apache.sysds.runtime.util;

import org.apache.sysds.common.Types;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.lineage.LineageRecomputeUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AutoDiff {
	private static List<Data> getAffineBackward( MatrixObject dout, MatrixObject X, MatrixObject W,
		ExecutionContext adec) {

		final String AD_VARDX = "__adx";
		final String AD_VARDW = "__adw";
		final String AD_VARDB = "__adb";

		adec.setVariable("X", X);
		adec.setVariable("dout", dout);
		adec.setVariable("W", W);
		DataOp input = HopRewriteUtils.createTransientRead("X", X);
		DataOp output = HopRewriteUtils.createTransientRead("dout", dout);
		DataOp w = HopRewriteUtils.createTransientRead("W", W);

		//dx = dout %*% t(w1)
		ReorgOp trasnW = HopRewriteUtils.createTranspose(w);
		AggBinaryOp matMulX = HopRewriteUtils.createMatrixMultiply(output, trasnW);
		DataOp dX = HopRewriteUtils.createTransientWrite(AD_VARDX, matMulX);
		//dw = t(X) %*% dout
		ReorgOp trasnX = HopRewriteUtils.createTranspose(input);
		AggBinaryOp matMulW = HopRewriteUtils.createMatrixMultiply(trasnX, output);
		DataOp dW = HopRewriteUtils.createTransientWrite(AD_VARDW, matMulW);
		// db = colSums(bias)
		AggUnaryOp colSum = HopRewriteUtils.createAggUnaryOp(output, Types.AggOp.SUM, Types.Direction.Col);
		DataOp dB = HopRewriteUtils.createTransientWrite(AD_VARDB, colSum);
		// create dX instruction
		ArrayList<Instruction> dXInst = Recompiler
			.recompileHopsDag(dX, adec.getVariables(), null, true, true, 0);
		// create dW instruction
		ArrayList<Instruction> dWInst = Recompiler
			.recompileHopsDag(dW, adec.getVariables(), null, true, true, 0);
		// create db instruction
		ArrayList<Instruction> dBInst = Recompiler
			.recompileHopsDag(dB, adec.getVariables(), null, true, true, 0);
		executeInst(dXInst, adec);
		executeInst(dWInst, adec);
		executeInst(dBInst, adec);

		List<Data> outputList = new ArrayList<>();
		outputList.add(adec.getVariable(AD_VARDX));
		outputList.add(adec.getVariable(AD_VARDW));
		outputList.add(adec.getVariable(AD_VARDB));

		return outputList;
	}

	private static void executeInst (ArrayList<Instruction> newInst, ExecutionContext lrwec)
	{
		try {
			//execute instructions
			BasicProgramBlock pb = new BasicProgramBlock(new Program());
			pb.setInstructions(newInst);
			pb.execute(lrwec);
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Error executing autoDiff instruction" , e);
		}
	}

	public static ListObject getBackward(ArrayList<Data> forwardLayers, ArrayList<Data> lineage, ExecutionContext adec) {
		List<Data> data = new ArrayList<Data>();
		List<String> names = new ArrayList<String>();
		// reverse the list to start from the last layer
		Collections.reverse(forwardLayers);
		Collections.reverse(lineage);
		for(int i=0; i<forwardLayers.size(); i++)
		{
			// get the name
			String layer = forwardLayers.get(i).toString();
			if(layer.equals("affine"))
			{
				// get the lineage
				String lin = lineage.get(i).toString();
				// get rid of foo flag
				lin = lin.replace("foo", "");

				// get lineage of X, Y, W and dout for affine backward
				String[] linInst = lin.split("\n");
				// assumption first line in the lineage is X and second is weight
				MatrixObject dout = (MatrixObject)LineageRecomputeUtils.parseNComputeLineageTrace(lin, null);
				MatrixObject X =  (MatrixObject)LineageRecomputeUtils.parseNComputeLineageTrace(linInst[0], null);
				MatrixObject W =  (MatrixObject)LineageRecomputeUtils.parseNComputeLineageTrace(linInst[1], null);
				// prepare the named list
				names.add("dX");
				names.add("dW");
				names.add("dB");
				data = getAffineBackward(dout, X, W, adec);
			}
		}
		return new ListObject(data, names);
	}
}
