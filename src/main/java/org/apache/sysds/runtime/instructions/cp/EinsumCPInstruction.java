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
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.codegen.cplan.CNodeCell;
import org.apache.sysds.hops.codegen.cplan.CNodeRow;
import org.apache.sysds.runtime.codegen.*;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.functionobjects.*;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.LineageCodegenItem;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageItemUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.Operator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;

import java.util.*;

import static org.apache.sysds.runtime.instructions.cp.EinsumContext.getEinsumContext;

public class EinsumCPInstruction extends BuiltinNaryCPInstruction {

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

//	public static EinsumCPInstruction parseInstruction(String str) {
//		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
//
//		ArrayList<CPOperand> inlist = new ArrayList<>();
//
//		for (int i = 2; i < parts.length - 1; i++)
//		{
//			inlist.add(new CPOperand(parts[i]));
//		}
//
//		CPOperand out = new CPOperand(parts[parts.length-1]);
////		int k = 1;//Integer.parseInt(parts[parts.length-1]);
//		int k = OptimizerUtils.getConstrainedNumThreads(-1);
//
//		String eqString = new CPOperand(parts[1]).getName(); //todo change
//
//		return new EinsumCPInstruction(k, inlist.toArray(new CPOperand[0]), out, parts[0], str, eqString);
//	}

	@Override
	public void processInstruction(ExecutionContext ec) {

		//get input matrices and scalars, incl pinning of matrices
		ArrayList<MatrixBlock> inputs = new ArrayList<>();
		ArrayList<ScalarObject> scalars = new ArrayList<>();
		if( LOG.isDebugEnabled() )
			LOG.debug("executing spoof instruction " + _op);
		for (CPOperand input : _in) {
			if(input.getDataType()==DataType.MATRIX){
				MatrixBlock mb = ec.getMatrixInput(input.getName());
				//FIXME fused codegen operators already support compressed main inputs 
				if(mb instanceof CompressedMatrixBlock){
					mb = ((CompressedMatrixBlock) mb).getUncompressed("Spoof instruction");
				}
				inputs.add(mb);
			}
//			else if(input.getDataType()==DataType.SCALAR) {
//				//note: even if literal, it might be compiled as scalar placeholder
//				scalars.add(ec.getScalarInput(input));
//			}
		}

		EinsumContext einc = getEinsumContext(eqStr,inputs);


		String[] parts = einc.equationString.split("->");
		String[] inputsChars = parts[0].split(",");

//		System.out.println("outrows:"+einc.outRows);
//		System.out.println("outcols:"+einc.outCols);

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


			if(inputsChars.length == 2 && inputsChars[0].charAt(0)==inputsChars[1].charAt(0) && !einc.summingChars.contains(parts[1].charAt(0))){// ja,jb->...
				// outer tmpl
				CNodeRow cnode = new CNodeRow(new ArrayList<>(), null);
//				cnode.setConstDim2(einc.outCols);
//				cnode.setNumVectorIntermediates(1);
				String src = tmpRow;

				if(einc.outCols == 1){
//					cnode.setRowType(SpoofRowwise.RowType.ROW_AGG);
					src = src.replace("%TYPE%","ROW_AGG");

				}else {
//					cnode.setRowType(SpoofRowwise.RowType.COL_AGG_B1_T);
					src = src.replace("%TYPE%","COL_AGG_B1_T");

				}
				src = src.replace("%TMP%", cnode.createVarname());

//				String src= cnode.codegenEinsum(false, SpoofCompiler.GeneratorAPI.JAVA);
				src = src.replace("%CONST_DIM2%",einc.outCols.toString());// super(RowType.%TYPE%, %CONST_DIM2%, %TB1%, %VECT_MEM%);\n" +

//				System.out.println(src);
				Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);
//				Class cla = CodegenUtils.compileClass("codegen.TMP0", src);
				SpoofOperator op = CodegenUtils.createInstance(cla);
				MatrixBlock mb = new MatrixBlock();
				mb.reset(einc.outRows, einc.outCols, false);
				mb.allocateDenseBlock();
				if(LibSpoofPrimitives.isFlipOuter(einc.outRows,einc.outCols)){
//					System.out.println("swapping");
					ArrayList<MatrixBlock> m2 = new ArrayList<MatrixBlock>(2);

					m2.add(inputs.get(1));
					m2.add(inputs.get(0));
					MatrixBlock out =op.execute(m2,scalars,mb,_numThreads);
					ec.setMatrixOutput(output.getName(), out);
				}else{
//					System.out.println("NOTswapping");
					MatrixBlock out =op.execute(inputs,scalars,mb,_numThreads);
					ec.setMatrixOutput(output.getName(), out);
				}

			}
			else if(inputsChars.length == 2 && inputsChars[0].charAt(1)==inputsChars[1].charAt(0)){
				ReorgOperator transpose =  new ReorgOperator(SwapIndex.getSwapIndexFnObject(), _numThreads);//todo move to separate op earlier
				MatrixBlock first = (inputs.get(0)).reorgOperations(transpose, new MatrixBlock(), 0 ,0, 0);

				CNodeRow cnode = new CNodeRow(new ArrayList<>(), null);
				String src = tmpRow;

				if(einc.outCols == 1){
					src = src.replace("%TYPE%","ROW_AGG");

				}else {
					src = src.replace("%TYPE%","COL_AGG_B1_T");

				}
				src = src.replace("%TMP%", cnode.createVarname());

				src = src.replace("%CONST_DIM2%",einc.outCols.toString());// super(RowType.%TYPE%, %CONST_DIM2%, %TB1%, %VECT_MEM%);\n" +

//				System.out.println(src);
				Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);
				SpoofOperator op = CodegenUtils.createInstance(cla);
				MatrixBlock mb = new MatrixBlock();
				mb.reset(einc.outRows, einc.outCols, false);

				mb.allocateDenseBlock();
				if(LibSpoofPrimitives.isFlipOuter(einc.outRows,einc.outCols)){
					ArrayList<MatrixBlock> m2 = new ArrayList<MatrixBlock>(2);

					m2.add(inputs.get(1));
					m2.add(first);
					MatrixBlock out =op.execute(m2,scalars,mb,_numThreads);
					ec.setMatrixOutput(output.getName(), out);
				}else{
					ArrayList<MatrixBlock> m2 = new ArrayList<MatrixBlock>(2);
					m2.add(first);
					m2.add(inputs.get(1));
					MatrixBlock out =op.execute(m2,scalars,mb,_numThreads);
					ec.setMatrixOutput(output.getName(), out);
				}
			}
			else{ //fallback to cell
				CNodeCell cnode = new CNodeCell(new ArrayList<>(), null);
//				cnode.setCellType(SpoofCellwise.CellType.NO_AGG);
				StringBuilder sb = new StringBuilder();

				String outputChars = parts[1];
				if( outputChars.length()==2){
					einc.summingChars.remove( outputChars.charAt(0));
					einc.summingChars.remove( outputChars.charAt(1));

				}else if( outputChars.length()==1){
					einc.summingChars.remove( outputChars.charAt(0));

				}
				boolean needsSumming = einc.summingChars.stream().anyMatch(x->x != null);
				String itVar0 = "TMP123";//+new IDSequence().getNextID(); todo: generate this var
				String outVar = null;
				if(needsSumming){
					outVar = "TMP123";//+new IDSequence().getNextID();
					sb.append("double ");
					sb.append(outVar);
					sb.append("=0;\n");
				}

				HashSet<Character> summedCharacters = new HashSet<>();
				Iterator<Character> hsIt = einc.summingChars.iterator();
				while (hsIt.hasNext()) {

					Character c = hsIt.next();
					String itVar = itVar0+c;
					sb.append("for(int ");
					sb.append(itVar);
					sb.append("=0;");
					sb.append(itVar);
					sb.append("<");
					sb.append(einc.charToDimensionSizeInt.get(c));
					sb.append(";");
					sb.append(itVar);
					sb.append("++){\n");
				}
				if (needsSumming){
					sb.append(outVar);
					sb.append("+=");
				}

				if(parts[1].length()==2){
					for (int i=0;i< inputsChars.length;i++){
						if(einc.summingChars.contains(inputsChars[i].charAt(0))){
							sb.append("getValue(b[");
							sb.append(i);
							sb.append("],b[");
							sb.append(i);
							sb.append("].clen,");
							sb.append(itVar0);
							sb.append(inputsChars[i].charAt(0));
							sb.append(",");
						}else if(inputsChars[i].charAt(0)==outputChars.charAt(0)) {
							sb.append("getValue(b[");
							sb.append(i);
							sb.append("],b[");
							sb.append(i);
							sb.append("].clen, rix,");
						}else if(inputsChars[i].charAt(0)==outputChars.charAt(1)) {
							sb.append("getValue(b[");
							sb.append(i);
							sb.append("],b[");
							sb.append(i);
							sb.append("].clen, cix,");
						}else {
							sb.append("getValue(b[");
							sb.append(i);
							sb.append("],b[");
							sb.append(i);
							sb.append("].clen, 0,");
						}

						if(einc.summingChars.contains(inputsChars[i].charAt(1))) {
							sb.append(itVar0);
							sb.append(inputsChars[i].charAt(1));
							sb.append(")");
						}
						else if(inputsChars[i].charAt(1)==outputChars.charAt(0)){
								sb.append("rix)");
						}else if(inputsChars[i].charAt(1)==outputChars.charAt(1)){
							sb.append("cix)");

						}else {
							sb.append("0)");
						}


						if(i<inputsChars.length-1) {
							sb.append("*");
						}

					}
				}else{
					for (int i=0;i< inputsChars.length;i++){
						if(inputsChars.length==2){
							if(einc.summingChars.contains(inputsChars[i].charAt(0))){
								sb.append("getValue(b[");
								sb.append(i);
								sb.append("],b[");
								sb.append(i);
								sb.append("].clen,");
								sb.append(itVar0);
								sb.append(inputsChars[i].charAt(0));
								sb.append(",");
							}else if(inputsChars[i].charAt(0)==outputChars.charAt(0)) {
								sb.append("getValue(b[");
								sb.append(i);
								sb.append("],b[");
								sb.append(i);
								sb.append("].clen, rix,");
							}else if(inputsChars[i].charAt(0)==outputChars.charAt(1)) {
								sb.append("getValue(b[");
								sb.append(i);
								sb.append("],b[");
								sb.append(i);
								sb.append("].clen, cix,");
							}else {
								sb.append("getValue(b[");
								sb.append(i);
								sb.append("],b[");
								sb.append(i);
								sb.append("].clen, 0,");
							}
							sb.append("0)");

							if(i<inputsChars.length-1) {
								sb.append("*");
							}

						}
					}
				}
				if (needsSumming){
					sb.append(";");
				}
				for(int si = 0; si<einc.summingChars.size(); si++){
					sb.append("}\n");
				}
				String src = tmpCell;
				src = src.replace("%TMP%" ,cnode.createVarname());
				if(needsSumming) {
					src = src.replace("%BODY_dense%", sb.toString());
					src = src.replace("%OUT%", outVar);
				}
				else{
					src = src.replace("%BODY_dense%" ,"");
					src = src.replace("%OUT%" , sb.toString());
				}

//				String src = needsSumming ? cnode.codegenEinsum(false, SpoofCompiler.GeneratorAPI.JAVA, sb.toString(), outVar) : cnode.codegenEinsum(false, SpoofCompiler.GeneratorAPI.JAVA, "", sb.toString());
//				System.out.println(src);
				Class cla = CodegenUtils.compileClass("codegen." + cnode.getClassname(), src);
				SpoofOperator op = CodegenUtils.createInstance(cla);
				MatrixBlock resBlock = new MatrixBlock();
				resBlock.reset(einc.outRows,einc.outCols);
					inputs.add(0, resBlock);
					MatrixBlock out =op.execute(inputs,scalars,new MatrixBlock(),_numThreads);
					if(einc.outRows==1 && einc.outCols ==1){
						ec.setScalarOutput(output.getName(), new DoubleObject( out.get(0,0)));

					}
					else{
						ec.setMatrixOutput(output.getName(), out);

					}
				}



		// release input matrices
		for (CPOperand input : _in)
			if(input.getDataType()==DataType.MATRIX)
				ec.releaseMatrixInput(input.getName());
	}

//	@Override
//	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
//		//return the lineage item if already traced once
//		LineageItem li = ec.getLineage().get(output.getName());
//		if (li != null)
//			return Pair.of(output.getName(), li);
//
//		//read and deepcopy the corresponding lineage DAG (pre-codegen)
//		LineageItem LIroot = LineageCodegenItem.getCodegenLTrace(getOperatorClass().getName()).deepCopy();
//
//		//replace the placeholders with original instruction inputs.
//		LineageItemUtils.replaceDagLeaves(ec, LIroot, _in);
//
//		return Pair.of(output.getName(), LIroot);
//	}

	public CPOperand[] getInputs() {
		return _in;
	}

	private static final IDSequence _idSeqfn = new IDSequence();

	private final static String tmpRow = "package codegen;\n" +
			"import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofRowwise;\n" +
			"import org.apache.sysds.runtime.codegen.SpoofRowwise.RowType;\n" +
			"import org.apache.commons.math3.util.FastMath;\n" +
			"\n" +
			"public final class %TMP% extends SpoofRowwise { \n" +
			"  public %TMP%() {\n" +
			"    super(RowType.%TYPE%, %CONST_DIM2%, false, 1);\n" +
			"  }\n" +
			"  protected void genexec(double[] a, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int len, long grix, int rix) { \n" +
			"LibSpoofPrimitives.vectOuterMultAdd(a, b[0].values(rix), c, ai, b[0].pos(rix), 0, len, b[0].clen);  }\n" +
			"  protected void genexec(double[] avals, int[] aix, int ai, SideInput[] b, double[] scalars, double[] c, int ci, int alen, int len, long grix, int rix) { \n" +
			"  }\n" +
			"}\n";

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
}
