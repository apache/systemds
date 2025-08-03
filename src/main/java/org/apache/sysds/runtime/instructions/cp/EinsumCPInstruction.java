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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.cplan.CNode;
import org.apache.sysds.hops.codegen.cplan.CNodeBinary;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeData;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.runtime.codegen.*;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.einsum.EinsumContext;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import java.util.*;

public class EinsumCPInstruction extends BuiltinNaryCPInstruction {
	public static boolean FORCE_CELL_TPL = false;
	protected static final Log LOG = LogFactory.getLog(EinsumCPInstruction.class.getName());
	public String eqStr;
	private final int _numThreads;
	private final CPOperand[] _in;

	public EinsumCPInstruction(Operator op, String opcode, String istr, CPOperand out, CPOperand... inputs)
	{
		super(op, opcode, istr, out, inputs);
		_numThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		_in = inputs;
		this.eqStr = inputs[0].getName();
		Logger.getLogger(EinsumCPInstruction.class).setLevel(Level.TRACE);
	}

	private EinsumContext einc = null;

	@Override
	public void processInstruction(ExecutionContext ec) {
		//get input matrices and scalars, incl pinning of matrices
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		for (CPOperand input : _in) {
			if(input.getDataType()==DataType.MATRIX){
				MatrixBlock mb = ec.getMatrixInput(input.getName());
				if(mb instanceof CompressedMatrixBlock){
					mb = ((CompressedMatrixBlock) mb).getUncompressed("Spoof instruction");
				}
				inputs.add(mb);
			}
		}

		EinsumContext einc = EinsumContext.getEinsumContext(eqStr, inputs);

		this.einc = einc;
		String resultString = einc.outChar2 != null ? String.valueOf(einc.outChar1) + einc.outChar2 : einc.outChar1 != null ? String.valueOf(einc.outChar1) : "";

		if( LOG.isDebugEnabled() ) LOG.trace("outrows:"+einc.outRows+", outcols:"+einc.outCols);

		ArrayList<String> inputsChars = einc.newEquationStringInputsSplit;

		if(LOG.isTraceEnabled()) LOG.trace(String.join(",",einc.newEquationStringInputsSplit));

		contractDimensionsAndComputeDiagonals(einc, inputs);

		//make all vetors col vectors
		for(int i = 0; i < inputs.size(); i++){
			if(inputs.get(i) != null && inputsChars.get(i).length() == 1) EnsureMatrixBlockColumnVector(inputs.get(i));
		}

		if(LOG.isTraceEnabled()) for(Character c : einc.characterAppearanceIndexes.keySet()){
			ArrayList<Integer> a = einc.characterAppearanceIndexes.get(c);
			LOG.trace(c+" count= "+a.size());
		}

		// compute scalar by suming-all matrices:
		Double scalar = null;
		for(int i=0;i< inputs.size(); i++){
			String s = inputsChars.get(i);
			if(s.equals("")){
				MatrixBlock mb = inputs.get(i);
				if (scalar == null) scalar = mb.get(0,0);
				else scalar*= mb.get(0,0);
				inputs.set(i,null);
				inputsChars.set(i,null);
			}
		}

		if (scalar != null) {
			boolean appliedToSomeMatrix = false;
			for(int i = 0; i < inputs.size(); i++){
				if(inputs.get(i) != null){
					inputs.set(i, getScalarMultiplyMatrixBlock(inputs.get(i), scalar));
					appliedToSomeMatrix = true; break;
				}
			}
			if(!appliedToSomeMatrix){
				ec.setScalarOutput(output.getName(), new DoubleObject(scalar));
				releaseMatrixInputs(ec);
				return;
			}
		}

		boolean needToDoCellTemplate = FORCE_CELL_TPL ? true : generatePlanAndExecute(inputs, einc);

		if (!needToDoCellTemplate){
			//check if any operations to do that were not-output dimension summations:
			List<String> remStrings = inputsChars.stream()
					.filter(Objects::nonNull).toList();
			List<MatrixBlock> remMbs = inputs.stream()
					.filter(Objects::nonNull).toList();
			MatrixBlock res;
			if(remStrings.size() == 1) {
				String s = remStrings.get(0);
				if(s.equals(resultString)){
					res=remMbs.get(0);
				}else if(s.charAt(0) == s.charAt(1)) {
					// diagonal needed
					ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
					res= remMbs.get(0).reorgOperations(op, new MatrixBlock(),0,0,0);
				}else{
					//it has to be transpose: ab->ba
					ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
					res = remMbs.get(0).reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				}
			} else{
				// maybe the leftovers are i,j and result should be ij or ji -> outer multp.
				if(remStrings.size() == 2 && remStrings.get(0).length()==1 && remStrings.get(1).length()==1){
					MatrixBlock first;
					MatrixBlock second;

					if(remStrings.get(0).charAt(0) == einc.outChar1 && remStrings.get(1).charAt(0) == einc.outChar2){
						first = remMbs.get(0);
						second = remMbs.get(1);
					}else if(remStrings.get(0).charAt(0) == einc.outChar2 && remStrings.get(1).charAt(0) == einc.outChar1){
						first = remMbs.get(1);
						second = remMbs.get(0);
					}else{
						throw new RuntimeException("Einsum runtime error: left with 2 vectors that cannot produce final result "+remStrings.get(0)+" , "+remStrings.get(1)); // should not happen
					}
					if(first.getNumColumns() > 1){
						int r = first.getNumColumns();
						first.setNumRows(r);
						first.setNumColumns(1);
						first.getDenseBlock().resetNoFill(r,1);
					}
					if(second.getNumRows() > 1){
						int c = second.getNumRows();
						second.setNumRows(1);
						second.setNumColumns(c);
						second.getDenseBlock().resetNoFill(1,c);
					}
					res = LibMatrixMult.matrixMult(first,second, _numThreads);
				}else {
					throw new RuntimeException("Einsum runtime error, reductions and multiplications finished but the did not produce one result"); // should not happen
				}
			}
			if (einc.outRows == 1 && einc.outCols == 1)
				ec.setScalarOutput(output.getName(), new DoubleObject(res.get(0, 0)));
			else ec.setMatrixOutput(output.getName(), res);
		}

		else { // if (needToDoCellTemplate)
			ArrayList<MatrixBlock> mbs = new ArrayList<>();
			ArrayList<String> chars = new ArrayList<>();
			for (int i = 0; i < inputs.size(); i++) {
				MatrixBlock mb = inputs.get(i);
				if (mb != null) {
					mbs.add(mb);
					chars.add(inputsChars.get(i));
				}
			}
			ArrayList summingChars = new ArrayList();
			for (Character c : einc.characterAppearanceIndexes.keySet()) {
				if (c != einc.outChar1 && c != einc.outChar2) summingChars.add(c);
			}
			if(LOG.isTraceEnabled()) LOG.trace("finishing with cell tpl: "+String.join(",", chars));

			MatrixBlock res = computeCellSummation(mbs, chars, resultString, einc.charToDimensionSize, summingChars, einc.outRows, einc.outCols);

			if (einc.outRows == 1 && einc.outCols == 1)
				ec.setScalarOutput(output.getName(), new DoubleObject(res.get(0, 0)));
			else ec.setMatrixOutput(output.getName(), res);
		}

		if(LOG.isTraceEnabled()) LOG.trace("EinsumCPInstruction Finished");

		releaseMatrixInputs(ec);
	}

	private void contractDimensionsAndComputeDiagonals(EinsumContext einc, ArrayList<MatrixBlock> inputs) {
		for(int i = 0; i< einc.contractDims.length; i++){
			//AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(),Types.CorrectionLocationType.LASTCOLUMN);
			AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

			if(einc.diagonalInputs[i]){
				ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
				inputs.set(i, inputs.get(i).reorgOperations(op, new MatrixBlock(),0,0,0));
			}
			if (einc.contractDims[i] == null) continue;
			switch (einc.contractDims[i]){
				case CONTRACT_BOTH: {
					AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), _numThreads);
					MatrixBlock res = new MatrixBlock(1, 1, false);
					inputs.get(i).aggregateUnaryOperations(aggun, res, 0, null);
					inputs.set(i, res);
					break;
				}
				case CONTRACT_RIGHT: {
					AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), _numThreads);
					MatrixBlock res = new MatrixBlock(inputs.get(i).getNumRows(), 1, false);
					inputs.get(i).aggregateUnaryOperations(aggun, res, 0, null);
					inputs.set(i, res);
					break;
				}
				case CONTRACT_LEFT: {
					AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), _numThreads);
					MatrixBlock res = new MatrixBlock(inputs.get(i).getNumColumns(), 1, false);
					inputs.get(i).aggregateUnaryOperations(aggun, res, 0, null);
					inputs.set(i, res);
					break;
				}
				default:
					break;
			}
		}
	}

	private void releaseMatrixInputs(ExecutionContext ec){
		for (CPOperand input : _in)
			if(input.getDataType()==DataType.MATRIX)
				ec.releaseMatrixInput(input.getName());
	}

	// returns true if there are elements that appear more than 2 times and cannot be summed
	private boolean generatePlanAndExecute(ArrayList<MatrixBlock> inputs, EinsumContext einc) {
		boolean anyCouldNotDo;
		boolean didAnything = false; // maybe multiplication will make it summable
		do {
			anyCouldNotDo = sumCharactersWherePossible(einc.characterAppearanceIndexes, inputs, einc.newEquationStringInputsSplit, einc.outChar1, einc.outChar2);
			didAnything = false;
			if(einc.newEquationStringInputsSplit.stream().filter(Objects::nonNull).count() > 1)
				didAnything = multiplyTerms(einc.characterAppearanceIndexes, inputs, einc.newEquationStringInputsSplit, einc.outChar1, einc.outChar2);
		}
		while(didAnything);

		if(LOG.isTraceEnabled()) LOG.trace("generatePlanAndExecute() finished");

		return anyCouldNotDo;
	}

	private static void EnsureMatrixBlockColumnVector(MatrixBlock mb){
		if(mb.getNumColumns() > 1){
			mb.setNumRows(mb.getNumColumns());
			mb.setNumColumns(1);
			mb.getDenseBlock().resetNoFill(mb.getNumRows(),1);
		}
	}
	private static void EnsureMatrixBlockRowVector(MatrixBlock mb){
		if(mb.getNumRows() > 1){
			mb.setNumColumns(mb.getNumRows());
			mb.setNumRows(1);
			mb.getDenseBlock().resetNoFill(1,mb.getNumColumns());
		}
	}

	/*  handle situation: ji,ji or ij,ji, j,j */
	private boolean multiplyTerms(HashMap<Character, ArrayList<Integer>> partsCharactersToIndices, ArrayList<MatrixBlock> inputs, ArrayList<String> inputsChars, Character outChar1, Character outChar2 ) {
		HashMap<String, LinkedList<Integer>> matrixStringToIndex = new HashMap<>();
		HashSet<String> matrixStringToIndexSkip = new HashSet<>();
		HashMap<String, LinkedList<Integer>> vectorStringToIndex = new HashMap<>();

		for(int i = 0; i < inputsChars.size(); i++){
			String s = inputsChars.get(i);
			if(s==null) continue;

			if(s.length() == 1){
				if (vectorStringToIndex.containsKey(s)) vectorStringToIndex.get(s).add(i);
				else { LinkedList<Integer> list = new LinkedList<>(); list.add(i); vectorStringToIndex.put(s, list); }
			}else{
				if (matrixStringToIndex.containsKey(s)) matrixStringToIndex.get(s).add(i);
				else { LinkedList<Integer> list = new LinkedList<>(); list.add(i); matrixStringToIndex.put(s, list); }
			}
		}

		boolean doneAnything = false;

		// first do the vector-vector multipl.
		for(var s : vectorStringToIndex.keySet()){
			if(vectorStringToIndex.get(s).size() <= 1) continue;

			doneAnything = true;

			Integer vectorIdx0 = vectorStringToIndex.get(s).removeFirst();

			if(LOG.isTraceEnabled()){
				StringBuilder sb = new StringBuilder();

				LOG.trace("Element wise multiplying: "+s);
			}
			MatrixBlock mb = inputs.get(vectorIdx0);
			EnsureMatrixBlockColumnVector(mb);

			do{ // do all vectors with same char
				Integer vectorIdx1 = vectorStringToIndex.get(s).removeFirst();

				ArrayList<MatrixBlock> mbs = new ArrayList<>();

				mbs.add(mb);
				mbs.add(inputs.get(vectorIdx1));
				EnsureMatrixBlockColumnVector(mbs.get(1));

				mb = getCodegenElemwiseMult(mbs);

				inputs.set(vectorIdx1, null);
				inputsChars.set(vectorIdx1, null);
			} while (vectorStringToIndex.get(s).size() > 1);

			inputs.set(vectorIdx0, null);
			inputsChars.set(vectorIdx0, null);

			inputs.add(mb);
			inputsChars.add(s);
			if (partsCharactersToIndices.containsKey(s.charAt(0))) partsCharactersToIndices.get(s.charAt(0)).add(inputs.size() - 1);
			vectorStringToIndex.get(s).add(inputs.size() - 1);
		}

		for(var s : matrixStringToIndex.keySet()){
			if(matrixStringToIndexSkip.contains(s)) continue;

			String sT = s.length() == 2 ? String.valueOf(s.charAt(1)) + s.charAt(0) : null;
			LinkedList<Integer> idxs = matrixStringToIndex.get(s);
			LinkedList<Integer> idxsT = sT != null ? matrixStringToIndex.containsKey(sT) ? matrixStringToIndex.get(sT) : null : null;

			Integer vectorIdx0 = null;
			Integer vectorIdx1 = null;
			String char0 = String.valueOf(s.charAt(0));
			if(vectorStringToIndex.containsKey(char0)){
				// ab,a-> ab
				vectorIdx0 = vectorStringToIndex.get(char0).removeFirst(); // only one should left
			}

			String char1 = String.valueOf(s.charAt(1));
			if(vectorStringToIndex.containsKey(char1)){
				// ab,b -> ab
				vectorIdx1 = vectorStringToIndex.get(char1).removeFirst(); // only one should left
			}

			if(idxs.size() <= 1 && idxsT == null && vectorIdx0 == null && vectorIdx1 == null) continue;

			doneAnything = true;

			// do decision if transpose idxs or idxsT: right now just always transpose second
			if(LOG.isTraceEnabled()){
				StringBuilder sb = new StringBuilder();
				for(Integer idx : idxs){
					sb.append(inputsChars.get(idx));
					sb.append(",");
				}
				if(idxsT != null) for(Integer idx : idxsT){
					sb.append(inputsChars.get(idx));
					sb.append(",");
				}
				if(vectorIdx0 != null) {  sb.append(s.charAt(0)).append(","); }
				if(vectorIdx1 != null) {  sb.append(s.charAt(1)).append(","); }

				LOG.trace("Element wise multiplying: "+sb.toString());
			}
			Integer matrixIdx0 = idxs.removeFirst();

			MatrixBlock mb = inputs.get(matrixIdx0);

			inputs.set(matrixIdx0, null);
			inputsChars.set(matrixIdx0, null);

			while (!idxs.isEmpty()){
				Integer matrixIdx1 = idxs.removeFirst();

				ArrayList<MatrixBlock> mbs = new ArrayList<>();
				mbs.add(mb);
				mbs.add(inputs.get(matrixIdx1));

				mb = getCodegenElemwiseMult(mbs);

				inputs.set(matrixIdx1, null);
				inputsChars.set(matrixIdx1, null);
			}

			if(idxsT != null) while(!idxsT.isEmpty()) {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);

				Integer matrixIdx1 = idxsT.removeFirst();

				ArrayList<MatrixBlock> mbs = new ArrayList<>();
				mbs.add(mb);
				mbs.add(inputs.get(matrixIdx1).reorgOperations(transpose, new MatrixBlock(), 0, 0, 0));

				mb = getCodegenElemwiseMult(mbs);

				inputs.set(matrixIdx1, null);
				inputsChars.set(matrixIdx1, null);
			}

			if(vectorIdx1 != null){ // ab,b->ab
				EnsureMatrixBlockRowVector(inputs.get(vectorIdx1));

				mb = getRowCodegenMatrixBlock(mb, inputs.get(vectorIdx1), CNodeBinary.BinType.VECT_MULT, SpoofRowwise.RowType.NO_AGG,  null);

				inputs.set(vectorIdx1, null);
				inputsChars.set(vectorIdx1, null);
			}

			if(vectorIdx0 != null){ // ab,a->ab
				EnsureMatrixBlockRowVector(inputs.get(vectorIdx0));

//				mb = getRowCodegenMatrixBlock(mb, inputs.get(vectorIdx0), CNodeBinary.BinType.VECT_MULT, SpoofRowwise.RowType.NO_AGG,null);
//				mb = getRowCodegenMatrixBlock(mb, inputs.get(vectorIdx0), CNodeBinary.BinType.VECT_MULT_SCALAR, SpoofRowwise.RowType.NO_AGG, Long.valueOf( inputs.get(vectorIdx0).getNumColumns()));
				{
					ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
					mb = mb.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
					mb = getRowCodegenMatrixBlock(mb, inputs.get(vectorIdx0), CNodeBinary.BinType.VECT_MULT, SpoofRowwise.RowType.NO_AGG, null);
				}

				inputs.set(vectorIdx0, null);
				inputsChars.set(vectorIdx0, null);
				s = String.valueOf(s.charAt(1))+String.valueOf(s.charAt(0));
			}

			inputs.add(mb);
			inputsChars.add(s);
			for (int i = 0; i < s.length(); i++) { // for each char in string, add pointer to newly created entry
				char c = s.charAt(i);
				if (partsCharactersToIndices.containsKey(c)) partsCharactersToIndices.get(c).add(inputs.size() - 1);
			}

			if(idxsT != null) matrixStringToIndexSkip.add(sT);
		}



		return doneAnything;
	}

	// returns true if left with summation with more than 2 inputs
	private boolean sumCharactersWherePossible(HashMap<Character, ArrayList<Integer>> partsCharactersToIndices, ArrayList<MatrixBlock> inputs, ArrayList<String> inputsChars, Character outChar1, Character outChar2) {
		boolean anyCouldNotDo;

		while (true) {
			List<Integer> toSum = null;
			Character sumC = null;
			anyCouldNotDo = false;
			for (Character c : partsCharactersToIndices.keySet()) { // sum one dim at the time
				if (c == outChar1 || c == outChar2)
					continue;
				toSum = new ArrayList<>();
				for (Integer idx : partsCharactersToIndices.get(c).stream().filter(Objects::nonNull).toList()) {
					if (inputs.get(idx) != null) {
						toSum.add(idx);
					}
				}
				if (toSum.size() > 2) {
					anyCouldNotDo = true;
					continue;
				}
				if (toSum.size() != 2)
					continue;
				sumC = c;
				break;
			}

			if(sumC == null) break;

			Pair<MatrixBlock, String> res = computeRowSummation(toSum, inputs, inputsChars, sumC);
			String newS = res.getRight();

			for (Integer idx : toSum) {
				inputs.set(idx, null);
				inputsChars.set(idx, null);
			}
			inputs.add(res.getLeft());
			inputsChars.add(newS);

			for (int i = 0; i < newS.length(); i++) { // for each char in string, add pointer to newly created entry
				char c = newS.charAt(i);
				if(partsCharactersToIndices.containsKey(c))
					partsCharactersToIndices.get(c).add(inputs.size() - 1);
			}
			partsCharactersToIndices.remove(sumC);
		}
		return anyCouldNotDo;
	}

	private enum SumOperation {
		aB_a,
		Ba_a,
		Ba_aC, // mmult -> BC
//		aB_Ca,
		Ba_Ca,
		aB_aC, // outer mult
		a_a,
		aB_aB, Ba_Ba, Ba_aB, aB_Ba,// mult and sums, something like ij,ij->i
	}

	private Pair<MatrixBlock, String> computeRowSummation(List<Integer> toSum, ArrayList<MatrixBlock> inputs, List<String> inputsChars, Character sumChar) {

		if(toSum.size() != 2){
			return null;
		}

		String s1 = inputsChars.get(toSum.get(0));
		String s2 = inputsChars.get(toSum.get(1));

		MatrixBlock first = null;
		MatrixBlock second = null;

		String resS;
		SumOperation sumOp;

		if(s1.length()==1 && s2.length() == 1){
			sumOp = SumOperation.a_a;
			resS = "";
			first = inputs.get(toSum.get(0));
			second = inputs.get(toSum.get(1));
		}
		else if(s2.length() == 1 || s1.length() == 1){
			if(s1.length() == 1){
				String sTemp = s1;
				s1=s2;
				s2=sTemp;

				first = inputs.get(toSum.get(1));
				second = inputs.get(toSum.get(0));
			}else{
				first = inputs.get(toSum.get(0));
				second = inputs.get(toSum.get(1));
			}

			if(s1.charAt(0) == s2.charAt(0)){
				sumOp = SumOperation.aB_a;
				resS = String.valueOf(s1.charAt(1));
			}else{
				sumOp = SumOperation.Ba_a;
				resS = String.valueOf(s1.charAt(0));
			}
		} else if (s1.equals(s2)) {
			if(s1.charAt(0) == sumChar){
				sumOp = SumOperation.aB_aB;
				first = inputs.get(toSum.get(0));
				second = inputs.get(toSum.get(1));
				resS = String.valueOf(s1.charAt(1));
			}else{
				sumOp = SumOperation.Ba_Ba;
				first = inputs.get(toSum.get(0));
				second = inputs.get(toSum.get(1));
				resS = String.valueOf(s1.charAt(0));
			}
		}else if (s1.charAt(0) == s2.charAt(1) && s1.charAt(1) == s2.charAt(0)) {
			if(s1.charAt(0) == sumChar){
				sumOp = SumOperation.aB_Ba;
				first = inputs.get(toSum.get(0));
				second = inputs.get(toSum.get(1));
				resS = String.valueOf(s1.charAt(1));
			}else{
				sumOp = SumOperation.Ba_aB;
				first = inputs.get(toSum.get(0));
				second = inputs.get(toSum.get(1));
				resS = String.valueOf(s1.charAt(0));
			}
		} else if(s1.charAt(0) == s2.charAt(0)){
			sumOp = SumOperation.aB_aC;
			first = inputs.get(toSum.get(0));
			second = inputs.get(toSum.get(1));
			resS = String.valueOf(s1.charAt(1))+String.valueOf(s2.charAt(1));

		}
		else if(s1.charAt(1) == s2.charAt(1)){
			sumOp = SumOperation.Ba_Ca;
			first = inputs.get(toSum.get(0));
			second = inputs.get(toSum.get(1));
			resS = String.valueOf(s1.charAt(0))+String.valueOf(s2.charAt(0));
		}
		else if(s1.charAt(0) == s2.charAt(1)){
			sumOp = SumOperation.Ba_aC;
			String sTemp = s1;
			s1=s2;
			s2=sTemp;

			first = inputs.get(toSum.get(1));
			second = inputs.get(toSum.get(0));
			resS = String.valueOf(s1.charAt(0))+String.valueOf(s2.charAt(1));

		}
		else if(s1.charAt(1) == s2.charAt(0)){
			sumOp = SumOperation.Ba_aC;
			first = inputs.get(toSum.get(0));
			second = inputs.get(toSum.get(1));
			resS = String.valueOf(s1.charAt(0))+String.valueOf(s2.charAt(1));

		}else{
			throw new RuntimeException("Error when choosing row multiplication operation");
		}
		MatrixBlock out;

		if(LOG.isTraceEnabled()) LOG.trace("remaining: "+String.join(",",inputsChars.stream().filter(Objects::nonNull).toList()));
		if(LOG.isTraceEnabled()) LOG.trace("Summing: "+s1+","+s2+"->"+resS);
		switch (sumOp) {
			case aB_a:{
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				first = first.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.ROW_AGG,  null);
				break;
			}
			case Ba_a:
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.ROW_AGG,  null);
				break;
			case Ba_aC: {
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MATRIXMULT, SpoofRowwise.RowType.NO_AGG_B1,  Long.valueOf( second.getNumColumns()));
				break;
			}
			case Ba_Ca: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				second = second.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MATRIXMULT, SpoofRowwise.RowType.NO_AGG_B1, Long.valueOf(second.getNumColumns()));
				break;
			}
//			case aB_a:
			case aB_aC: {
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_OUTERMULT_ADD, SpoofRowwise.RowType.COL_AGG_B1_T, Long.valueOf( second.getNumColumns()));
				break;
			}
			case a_a:
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MULT, SpoofRowwise.RowType.NO_AGG,null);
				break;
			case aB_aB: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				first = first.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				second = second.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.COL_AGG, null);
				break;
			}
			case Ba_Ba: {
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.ROW_AGG, null);
				break;
			}
			case aB_Ba: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				first = first.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MATRIXMULT, SpoofRowwise.RowType.ROW_AGG,null);
				break;
			}
			case Ba_aB: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				second = second.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.ROW_AGG,null);
				break;
			}

			default:
				throw new IllegalStateException("Unexpected value: " + sumOp);
		}
		return Pair.of(out , resS);
	}

	private MatrixBlock getCodegenElemwiseMult(ArrayList<MatrixBlock> mbs) {

		ArrayList<CNode> cnodeIn = new ArrayList<>();
		for(int i=0;i<mbs.size(); i++){
			CNode c = new CNodeData("c"+i, i, mbs.get(i).getNumRows(), mbs.get(i).getNumColumns(), DataType.MATRIX);
			cnodeIn.add(c);
		}
		CNode cnodeOut = new CNodeBinary(cnodeIn.get(0), cnodeIn.get(1), CNodeBinary.BinType.VECT_MULT);

		CNodeRow cnode = new CNodeRow(cnodeIn, cnodeOut);

		cnode.setRowType(SpoofRowwise.RowType.NO_AGG);
		cnode.renameInputs();

		String src = cnode.codegen(false, SpoofCompiler.GeneratorAPI.JAVA);
		if( LOG.isTraceEnabled()) LOG.trace(CodegenUtils.printWithLineNumber(src));
		Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);

		SpoofOperator op = CodegenUtils.createInstance(cla);
		MatrixBlock mb = new MatrixBlock();

		ArrayList<ScalarObject> scalars = new ArrayList<>();
		MatrixBlock	out = op.execute(mbs, scalars, mb, _numThreads);
		return out;
	}
	private MatrixBlock getRowCodegenMatrixBlock(MatrixBlock first, MatrixBlock second, CNodeBinary.BinType binaryType, SpoofRowwise.RowType rowType, Long secondDim) {
		ArrayList<MatrixBlock> thisInputs = new ArrayList<>(Arrays.asList(first, second));

		ArrayList<CNode> cnodeIn = new ArrayList<>();

		CNode c1 = new CNodeData("c1", 1, first.getNumRows(), first.getNumColumns(), DataType.MATRIX);
		CNode c2 = new CNodeData("c2", 2, second.getNumRows(), second.getNumColumns(), DataType.MATRIX);
		cnodeIn.add(c1);
		cnodeIn.add(c2);
		CNode cnodeOut = new CNodeBinary(c1, c2, binaryType);
		CNodeRow cnode = new CNodeRow(cnodeIn, cnodeOut);

		cnode.setRowType(rowType);

		if(secondDim != null) cnode.setConstDim2(secondDim);
		cnode.renameInputs();

		String src = cnode.codegen(false, SpoofCompiler.GeneratorAPI.JAVA);
		if( LOG.isTraceEnabled()) LOG.trace(CodegenUtils.printWithLineNumber(src));
		Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);

		SpoofOperator op = CodegenUtils.createInstance(cla);
		MatrixBlock mb = new MatrixBlock();

		ArrayList<ScalarObject> scalars = new ArrayList<>();
	    MatrixBlock	out = op.execute(thisInputs, scalars, mb, _numThreads);
		return out;
	}

	private MatrixBlock getScalarMultiplyMatrixBlock(MatrixBlock mbIn, Double scalar){
		ArrayList<MatrixBlock> thisInputs = new ArrayList<>(Arrays.asList(mbIn));

		ArrayList<CNode> cnodeIn = new ArrayList<>();

		CNode c1 = new CNodeData("c1", 1, mbIn.getNumRows(), mbIn.getNumColumns(), DataType.MATRIX);
		CNode c2 = new CNodeData(new LiteralOp(scalar), 0, 0, DataType.SCALAR);
		cnodeIn.add(c1);
		cnodeIn.add(c2);

		CNode cnodeOut = new CNodeBinary(c1,c2, CNodeBinary.BinType.MULT);
		CNodeCell cnode = new CNodeCell(cnodeIn, cnodeOut);
		cnode.setCellType(SpoofCellwise.CellType.NO_AGG);
		cnode.renameInputs();

		String src = cnode.codegen(false, SpoofCompiler.GeneratorAPI.JAVA);
		if( LOG.isTraceEnabled()) LOG.trace(CodegenUtils.printWithLineNumber(src));
		Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);

		SpoofOperator op = CodegenUtils.createInstance(cla);
		MatrixBlock mb = new MatrixBlock();

		ArrayList<ScalarObject> scalars = new ArrayList<>();
		if(scalar != null) scalars.add(new DoubleObject(scalar));
		MatrixBlock	out = op.execute(thisInputs, scalars, mb, _numThreads);
		return out;
	}
	private static void indent(StringBuilder sb, int level) {
		for (int i = 0; i < level; i++) {
			sb.append("  ");
		}
	}

	private MatrixBlock computeCellSummation(ArrayList<MatrixBlock> inputs, List<String> inputsChars, String resultString,
														   HashMap<Character, Integer> charToDimensionSizeInt, List<Character> summingChars, int outRows, int outCols){
		ArrayList<CNode> cnodeIn = new ArrayList<>();
		cnodeIn.add(new CNodeData(new LiteralOp(3), 0, 0, DataType.SCALAR));
		CNodeCell cnode = new CNodeCell(cnodeIn, null);
		StringBuilder sb = new StringBuilder();

		int indent = 2;
		indent(sb, indent);

		boolean needsSumming = summingChars.stream().anyMatch(x -> x != null);

		String itVar0 = cnode.createVarname();
		String outVar = itVar0;
		if (needsSumming) {
			sb.append("double ");
			sb.append(outVar);
			sb.append("=0;\n");
		}

		HashSet<Character> summedCharacters = new HashSet<>();
		Iterator<Character> hsIt = summingChars.iterator();
		while (hsIt.hasNext()) {
			indent(sb, indent);
			indent++;
			Character c = hsIt.next();
			String itVar = itVar0 + c;
			sb.append("for(int ");
			sb.append(itVar);
			sb.append("=0;");
			sb.append(itVar);
			sb.append("<");
			sb.append(charToDimensionSizeInt.get(c));
			sb.append(";");
			sb.append(itVar);
			sb.append("++){\n");
		}
		indent(sb, indent);
		if (needsSumming) {
			sb.append(outVar);
			sb.append("+=");
		}

		for (int i = 0; i < inputsChars.size(); i++) {
			if (inputsChars.get(i).length() == 0){
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, 0,");
			}

			else if (summingChars.contains(inputsChars.get(i).charAt(0))) {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen,");
				sb.append(itVar0);
				sb.append(inputsChars.get(i).charAt(0));
				sb.append(",");
			} else if (resultString.length() >= 1  && inputsChars.get(i).charAt(0) == resultString.charAt(0)) {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, rix,");
			} else if (resultString.length() == 2 && inputsChars.get(i).charAt(0) == resultString.charAt(1)) {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, cix,");
			} else {
				sb.append("getValue(b[");
				sb.append(i);
				sb.append("],b[");
				sb.append(i);
				sb.append("].clen, 0,");
			}

			if (inputsChars.get(i).length() != 2){
				sb.append("0)");
			}
			else if (summingChars.contains(inputsChars.get(i).charAt(1))) {
				sb.append(itVar0);
				sb.append(inputsChars.get(i).charAt(1));
				sb.append(")");
			} else if (resultString.length() >= 1 &&inputsChars.get(i).charAt(1) == resultString.charAt(0)) {
				sb.append("rix)");
			} else if (resultString.length() == 2  && inputsChars.get(i).charAt(1) == resultString.charAt(1)) {
				sb.append("cix)");
			} else {
				sb.append("0)");
			}

			if (i < inputsChars.size() - 1) {
				sb.append(" * ");
			}

		}
		if (needsSumming) {
			sb.append(";\n");
		}
		indent--;
		for (int si = 0; si < summingChars.size(); si++) {
			indent(sb, indent);
			indent--;
			sb.append("}\n");
		}
		String src = CNodeCell.JAVA_TEMPLATE;//
		src = src.replace("%TMP%", cnode.createVarname());
		src = src.replace("%TYPE%", "NO_AGG");
		src = src.replace("%SPARSE_SAFE%", "false");
		src = src.replace("%SEQ%", "true");
		src = src.replace("%AGG_OP_NAME%", "null");
		if (needsSumming) {
			src = src.replace("%BODY_dense%", sb.toString());
			src = src.replace("%OUT%", outVar);
		} else {
			src = src.replace("%BODY_dense%", "");
			src = src.replace("%OUT%", sb.toString());
		}

		LOG.trace(src);
		Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);
		SpoofOperator op = CodegenUtils.createInstance(cla);
		MatrixBlock resBlock = new MatrixBlock();
		resBlock.reset(outRows, outCols);
		inputs.add(0, resBlock);
		MatrixBlock out = op.execute(inputs, new ArrayList<>(), new MatrixBlock(), _numThreads);

		return out;
	}

	public CPOperand[] getInputs() {
		return _in;
	}
}
