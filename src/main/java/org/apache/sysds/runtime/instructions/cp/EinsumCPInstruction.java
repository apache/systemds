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

import org.apache.commons.lang3.NotImplementedException;
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
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import java.util.*;

import static org.apache.sysds.runtime.instructions.cp.EinsumContext.getEinsumContext;

public class EinsumCPInstruction extends BuiltinNaryCPInstruction {
	public static boolean FORCE_CELL_TPL = false;
	protected static final Log LOG = LogFactory.getLog(EinsumCPInstruction.class.getName());
	public String eqStr;
	private final Class<?> _class;
	private final SpoofOperator _op;
	private final int _numThreads;
	private final CPOperand[] _in;

	public EinsumCPInstruction(Operator op, String opcode, String istr, CPOperand out, CPOperand... inputs)
	{
		super(op, opcode, istr, out, inputs);
		_class = null;
		_op = null;
		_numThreads = OptimizerUtils.getConstrainedNumThreads(-1);
		_in = inputs;
		this.eqStr = inputs[0].getName();
		Logger.getLogger(EinsumCPInstruction.class).setLevel(Level.TRACE);
	}


	public SpoofOperator getSpoofOperator() {
		return _op;
	}

	public Class<?> getOperatorClass() {
		return _class;
	}

	private static final int CONTRACT_LEFT = 1;
	private static final int CONTRACT_RIGHT = 2;
	private static final int CONTRACT_BOTH = 3;

	private EinsumContext einc = null;

	@Override
	public void processInstruction(ExecutionContext ec) {

		//get input matrices and scalars, incl pinning of matrices
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		ArrayList<String> inputsNames = new ArrayList<>();
		for (CPOperand input : _in) {
			if(input.getDataType()==DataType.MATRIX){
				MatrixBlock mb = ec.getMatrixInput(input.getName());
				//FIXME fused codegen operators already support compressed main inputs 
				if(mb instanceof CompressedMatrixBlock){
					mb = ((CompressedMatrixBlock) mb).getUncompressed("Spoof instruction");
				}
				inputs.add(mb);
				inputsNames.add(input.getName());
			}
		}

		EinsumContext einc = getEinsumContext(eqStr,inputs);
		this.einc = einc;
 //todo not throwing err when output char is not in input

		String[] parts = einc.equationString.split("->");
//		ArrayList<String> inputsChars = new ArrayList<>(Arrays.asList(parts[0].split(",")));

		if( LOG.isDebugEnabled() ) LOG.trace("outrows:"+einc.outRows+", outcols:"+einc.outCols);

		Character outChar1 = null;
		Character outChar2 = null;

		if(parts[1].length()>=2){
			outChar1 = parts[1].charAt(0);
			outChar2 = parts[1].charAt(1);
		}else if (parts[1].length()==1){
			outChar1 = parts[1].charAt(0);
		}
		HashMap<Character, ArrayList<Integer>> partsCharactersToIndices = new HashMap<>();
		ArrayList<String> newEquationStringSplit = new ArrayList();

		ArrayList<Integer> diagMatrices = new ArrayList<>();
		int arrCounter=0;
		for(int i=0;i<parts[0].length(); i++){
			char c  =parts[0].charAt(i);
			if(c==','){
				arrCounter++;
				continue;
			}
			String s="";
			if(!einc.contractDimsSet.contains(c)){

				if(!partsCharactersToIndices.containsKey(c))
					partsCharactersToIndices.put(c, new ArrayList<>());

				partsCharactersToIndices.get(c).add(arrCounter);
				s+=c;
			}
			if(i+1<parts[0].length()){
				char c2 = parts[0].charAt(i+1);
				if(c2==','){
					arrCounter++;
				}
				else if(!einc.contractDimsSet.contains(c2)){
					if (c2==c ){
						diagMatrices.add(arrCounter);
					}

//					if(partsCharactersCounter.containsKey(c2))
//						partsCharactersCounter.put(c2, partsCharactersCounter.get(c2)+1);
//					else partsCharactersCounter.put(c2, 1);

					if(!partsCharactersToIndices.containsKey(c2))
						partsCharactersToIndices.put(c2, new ArrayList<>());

					partsCharactersToIndices.get(c2).add(arrCounter);
					s+=c2;
				}

				i++;

			}
			newEquationStringSplit.add(s);
		}
		ArrayList<String> inputsChars = newEquationStringSplit;
		LOG.trace(String.join(",",newEquationStringSplit));
		//todo move to separate op earlier:
		for(int i=0;i<einc.contractDims.length; i++){
			if(einc.contractDims[i] == null){

			}else if(einc.contractDims[i] == CONTRACT_BOTH) {
 				//sum all
//				AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(),Types.CorrectionLocationType.LASTCOLUMN);
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceAll.getReduceAllFnObject(), _numThreads);
				MatrixBlock newB = (MatrixBlock)inputs.get(i).aggregateUnaryOperations(aggun,new MatrixBlock(),inputs.get(i).getNumRows(),null);
				inputs.set(i, newB);

			}else if(einc.contractDims[i] == CONTRACT_RIGHT){
				//rowSums (remove 2nd dim)
//				AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), Types.CorrectionLocationType.LASTCOLUMN);
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceCol.getReduceColFnObject(), _numThreads);
				MatrixBlock newB = (MatrixBlock)inputs.get(i).aggregateUnaryOperations(aggun,new MatrixBlock(),inputs.get(i).getNumRows(),null);
				inputs.set(i, newB);

			}else if(einc.contractDims[i] == CONTRACT_LEFT){
				//colSums (remove 1st dim)
//				AggregateOperator agg = new AggregateOperator(0, KahanPlus.getKahanPlusFnObject(), Types.CorrectionLocationType.LASTROW);
				AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());

				AggregateUnaryOperator aggun = new AggregateUnaryOperator(agg, ReduceRow.getReduceRowFnObject(), _numThreads);
				MatrixBlock newB = (MatrixBlock)inputs.get(i).aggregateUnaryOperations(aggun,new MatrixBlock(),inputs.get(i).getNumColumns(),null);
				inputs.set(i, newB);
			}
		}
		for(Integer idx : diagMatrices){
			ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
			MatrixBlock mb = inputs.get(idx);
			inputs.set(idx, mb.reorgOperations(op, new MatrixBlock(),0,0,0));
		}


		if(LOG.isTraceEnabled())
			for(Character c :partsCharactersToIndices.keySet()){
				ArrayList<Integer> a = partsCharactersToIndices.get(c);
				LOG.trace(c+" count= "+a.size());
			}



		Double scalar = null;
		boolean anyCouldNotDo = FORCE_CELL_TPL ? true : generatePlanAndExecute(partsCharactersToIndices, inputs, inputsChars, outChar1, outChar2); // information to do cell tpl for remaining ones

		if (!anyCouldNotDo){
			//check if any operations to do that were not-output dimension summations:
			List<String> remStrings = inputsChars.stream()
					.filter(Objects::nonNull).toList();
			List<MatrixBlock> remMbs = inputs.stream()
					.filter(Objects::nonNull).toList();
			MatrixBlock res;
			if(remStrings.size() == 1){
				String s = remStrings.get(0);
				if(s.equals(parts[1])){
					res=remMbs.get(0);
				}else if(s.charAt(0)==s.charAt(1)) {
					// diagonal needed
					ReorgOperator op = new ReorgOperator(DiagIndex.getDiagIndexFnObject());
					res= remMbs.get(0).reorgOperations(op, new MatrixBlock(),0,0,0);
				}else{
					//it has to be transpose: ab->ba
					ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
					res = remMbs.get(0).reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				}
			}else{
				throw new RuntimeException("did not expect this!");
			}
			ec.setMatrixOutput(output.getName(), res);
		}

		else {
			ArrayList<MatrixBlock> mbs = new ArrayList<>();
			ArrayList<String> chars = new ArrayList<>();
			for (int i = 0; i < inputs.size(); i++) {
				MatrixBlock mb = inputs.get(i);
				if (mb != null) {
					mbs.add(mb);
					chars.add(inputsChars.get(i));
				}
			}
			if(chars.size()==1 && chars.get(0).equals(parts[1])){ // maybe result is correct after all... (need to improve logic if cell is needed but for now just check if maybe all is OK)
				ec.setMatrixOutput(output.getName(), mbs.get(0));
			}else {
				ArrayList summingChars = new ArrayList();
				for (Character c : partsCharactersToIndices.keySet()) {
					if (c != outChar1 && c != outChar2) summingChars.add(c);
				}

				MatrixBlock res = computeCellSummation(mbs, chars, parts[1], einc.charToDimensionSizeInt, summingChars, einc.outRows, einc.outCols);

				if (einc.outRows == 1 && einc.outCols == 1)
					ec.setScalarOutput(output.getName(), new DoubleObject(res.get(0, 0)));
				else ec.setMatrixOutput(output.getName(), res);
			}
		}

		//final operation


		// release input matrices
		for (CPOperand input : _in)
			if(input.getDataType()==DataType.MATRIX)
				ec.releaseMatrixInput(input.getName());
	}

	private boolean generatePlanAndExecute(HashMap<Character, ArrayList<Integer>> partsCharactersToIndices, ArrayList<MatrixBlock> inputs, ArrayList<String> inputsChars, Character outChar1, Character outChar2) {
		// compute scalars:
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
		boolean anyCouldNotDo;
		boolean didAnything = false;
		do {
			anyCouldNotDo = sumCharactersWherePossible(partsCharactersToIndices, inputs, inputsChars, outChar1, outChar2);
			didAnything = false;
			if(inputsChars.stream().filter(Objects::nonNull).count() > 1)
				didAnything = multiplyTerms(partsCharactersToIndices, inputs, inputsChars, outChar1, outChar2);
		}
		while(didAnything);

		return anyCouldNotDo;
	}

	/*  handle situation: ji,ji or ij,ji */
	private boolean multiplyTerms(HashMap<Character, ArrayList<Integer>> partsCharactersToIndices, ArrayList<MatrixBlock> inputs, ArrayList<String> inputsChars, Character outChar1, Character outChar2 ) {
		ArrayList<Integer> multiplyIdxs = new ArrayList<>();
		ArrayList<Integer> transposeMultiplyIdxs = new ArrayList<>();

		HashMap<String, ArrayList<Integer>> stringToIndex = new HashMap<>();

		for(int i = 0; i < inputsChars.size(); i++){
			String s = inputsChars.get(i);
			if(s==null) continue;
//			if(s.length() != 2) continue;

			if (stringToIndex.containsKey(s)) stringToIndex.get(s).add(i);
			else { ArrayList<Integer> list = new ArrayList<>(); list.add(i); stringToIndex.put(s, list); }
		}

		boolean doneAnything = false;

		for(var s : stringToIndex.keySet()){
			if(!stringToIndex.containsKey(s)) continue; // entries can be removed

			String sT = s.length() == 2 ? String.valueOf(s.charAt(1)) + s.charAt(0) : null;
			ArrayList<Integer> idxs = stringToIndex.get(s);
			ArrayList<Integer> idxsT = sT != null ? stringToIndex.containsKey(sT) ? stringToIndex.get(sT) : null : null;

			if(idxs.size() <= 1 && idxsT == null) continue;

			doneAnything = true;

			// do decision if transpose idxs or idxsT: right now just alway transpose second
			ArrayList<MatrixBlock> mbs = new ArrayList<>();
			if(LOG.isTraceEnabled()){
				StringBuilder sb = new StringBuilder();
				for(Integer idx : idxs){
					sb.append(inputsChars.get(idx));
					sb.append(",");
				}
				if(idxsT != null)
				for(Integer idx : idxsT){
					sb.append(inputsChars.get(idx));
					sb.append(",");
				}
				LOG.trace("Element wise multiplying: "+sb.toString());
			}
			for(Integer idx : idxs){
				mbs.add(inputs.get(idx));
				inputs.set(idx, null);
				inputsChars.set(idx, null);
			}
			ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
			if(idxsT != null)
			for(Integer idx : idxsT){
				mbs.add(inputs.get(idx).reorgOperations(transpose, new MatrixBlock(), 0, 0, 0));
				inputs.set(idx, null);
				inputsChars.set(idx, null);
			}

			MatrixBlock mb = getCodegenElemwiseMult(mbs);

			inputs.add(mb);
			inputsChars.add(s);

			for (int i = 0; i < s.length(); i++) { // for each char in string, add pointer to newly created entry
				char c = s.charAt(i);
				partsCharactersToIndices.get(c).add(inputs.size() - 1);
			}

			if(idxsT != null)
			stringToIndex.remove(sT);
		}

		return doneAnything;
	}

	private boolean sumCharactersWherePossible(HashMap<Character, ArrayList<Integer>> partsCharactersToIndices, ArrayList<MatrixBlock> inputs, ArrayList<String> inputsChars, Character outChar1, Character outChar2) {
		boolean anyCouldNotDo = false;

		while (true) {
			List<Integer> toSum = null;
			Character sumC = null;
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

	private enum AggregateAtEnd{
		Left,
		Right,
		Both,
		None,
	}
	private Pair<MatrixBlock, String> computRowSummationsOutputCharsOnly(List<MatrixBlock> inputs, List<String> inputsChars, String resString, Double scalar ){
		if(resString.length() == 1){
			// dont expect more than two of these, throw error if happens
			if(inputs.size() != 2) throw new RuntimeException("did not expects this, please fix me");
			MatrixBlock mb = getRowCodegenMatrixBlock(inputs.get(0), inputs.get(1), CNodeBinary.BinType.VECT_MULT, SpoofRowwise.RowType.NO_AGG);
			return Pair.of(mb, inputsChars.get(0));
		}else{ // resString.length() == 2
			// something like a,a,b,b,ab,ba
			// group them

			ArrayList<MatrixBlock> a = new ArrayList<>();
			ArrayList<MatrixBlock> b = new ArrayList<>();
			ArrayList<MatrixBlock> ab = new ArrayList<>();
			ArrayList<MatrixBlock> ba = new ArrayList<>();
			for(int i =0;i< inputs.size(); i++){
				String s = inputsChars.get(i);
				if(s.length() == 2){
					if(s.equals(resString)) ab.add(inputs.get(i));
					else ba.add(inputs.get(i));
				}else{
					if(s.charAt(0)==resString.charAt(0)) a.add(inputs.get(i));
					else b.add(inputs.get(i));
				}
			}
			// mult all a-s:
			// mult all b-s:
			// check if there is ab or ba
			// if no:
			//   then do outer product axb or bxa
			// if there is then:
			//   mult ba and a
			//   mult ab and b
			//   transp ba into ab
			//   mult 2 ab and ab
			return Pair.of( ab.get(0) ,resString);
//			throw new NotImplementedException("todo");
//			return null;
		}
	}
	private Pair<MatrixBlock, String> computeRowSummation(List<Integer> toSum, ArrayList<MatrixBlock> inputs, List<String> inputsChars, Character sumChar) {
		return computeRowSummation(toSum,inputs,inputsChars, null, sumChar);
	}
	private Pair<MatrixBlock, String> computeRowSummation(List<Integer> toSum, ArrayList<MatrixBlock> inputs, List<String> inputsChars, Double scalar, Character sumChar) {

		if(toSum.size() != 2){
			return null;
		}

		String s1 = inputsChars.get(toSum.get(0));
		String s2 = inputsChars.get(toSum.get(1));

		MatrixBlock first = null;
		MatrixBlock second = null;

		String resS;
		SumOperation sumOp;

		if(s1.length()==1 && s2.length() == 1){ //remove never happening here
			sumOp = SumOperation.a_a;
			resS = "";
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

		if(LOG.isTraceEnabled()) LOG.trace("Summing: "+s1+","+s2+"->"+resS);
		switch (sumOp) {
			case Ba_a:
				throw new NotImplementedException();
			case Ba_aC: {
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MATRIXMULT, SpoofRowwise.RowType.NO_AGG_B1,  Long.valueOf( second.getNumColumns()), scalar);
				break;
			}
			case Ba_Ca: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				second = second.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MATRIXMULT, SpoofRowwise.RowType.NO_AGG_B1, Long.valueOf(second.getNumColumns()), scalar);
				break;
			}
			case aB_a:
			case aB_aC: {
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_OUTERMULT_ADD, SpoofRowwise.RowType.COL_AGG_B1_T, Long.valueOf( second.getNumColumns()),scalar);
				break;
			}
			case a_a:
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MULT, SpoofRowwise.RowType.NO_AGG,null, scalar);
				break;
			case aB_aB: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				first = first.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				second = second.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.COL_AGG, null, scalar);
				break;
			}
			case Ba_Ba: {
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.ROW_AGG, null, scalar);
				break;
			}
			case aB_Ba: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				first = first.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.VECT_MATRIXMULT, SpoofRowwise.RowType.ROW_AGG,null, scalar);
				break;
			}
			case Ba_aB: {
				ReorgOperator transpose = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);
				second = second.reorgOperations(transpose, new MatrixBlock(), 0, 0, 0);
				out = getRowCodegenMatrixBlock(first, second, CNodeBinary.BinType.DOT_PRODUCT, SpoofRowwise.RowType.ROW_AGG,null, scalar);
				break;
			}

			default:
				throw new IllegalStateException("Unexpected value: " + sumOp);
		}
		return Pair.of(out , resS);
	}
	private MatrixBlock getRowCodegenMatrixBlock(MatrixBlock first, MatrixBlock second, CNodeBinary.BinType binaryType, SpoofRowwise.RowType rowType){
		return getRowCodegenMatrixBlock(first, second, binaryType,rowType,null, null);
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
	private MatrixBlock getRowCodegenMatrixBlock(MatrixBlock first, MatrixBlock second, CNodeBinary.BinType binaryType, SpoofRowwise.RowType rowType, Long secondDim, Double scalar) {
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
		;
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

//				String src = needsSumming ? cnode.codegenEinsum(false, SpoofCompiler.GeneratorAPI.JAVA, sb.toString(), outVar) : cnode.codegenEinsum(false, SpoofCompiler.GeneratorAPI.JAVA, "", sb.toString());
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

	private static final IDSequence _idSeqfn = new IDSequence();

	private final static String tmpCell =
			"package codegen;\n" +
			"import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofCellwise;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofCellwise.AggOp;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;\n" +
			"import org.apache.commons.math3.util.FastMath;\n" +
			"public final class %TMP% extends SpoofCellwise {\n" +
			"  public %TMP%() {\n" +
			"    super(CellType.NO_AGG, false, true, null);\n" +
			"  }\n" +
			"  protected double genexec(double a, SideInput[] b, double[] scalars, int m, int n, long grix, int rix, int cix) { \n" +
			"    %BODY_dense%" +
			"    return %OUT%;\n" +
			"  }\n" +
			"}";
	public static final String JAVA_TEMPLATE =
			"package codegen;\n"
					+ "import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;\n"
					+ "import org.apache.sysds.runtime.codegen.SpoofCellwise;\n"
					+ "import org.apache.sysds.runtime.codegen.SpoofCellwise.AggOp;\n"
					+ "import org.apache.sysds.runtime.codegen.SpoofCellwise.CellType;\n"
					+ "import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;\n"
					+ "import org.apache.commons.math3.util.FastMath;\n"
					+ "\n"
					+ "public final class %TMP% extends SpoofCellwise {\n"
					+ "  public %TMP%() {\n"
					+ "    super(CellType.%TYPE%, %SPARSE_SAFE%, %SEQ%, %AGG_OP_NAME%);\n"
					+ "  }\n"
					+ "  protected double genexec(double a, SideInput[] b, double[] scalars, int m, int n, long grix, int rix, int cix) { \n"
					+ "%BODY_dense%"
					+ "    return %OUT%;\n"
					+ "  }\n"
					+ "}\n";
}
