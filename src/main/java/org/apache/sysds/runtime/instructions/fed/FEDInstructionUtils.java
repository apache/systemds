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

package org.apache.sysds.runtime.instructions.fed;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.codegen.SpoofCellwise;
import org.apache.sysds.runtime.codegen.SpoofMultiAggregate;
import org.apache.sysds.runtime.codegen.SpoofOuterProduct;
import org.apache.sysds.runtime.codegen.SpoofRowwise;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateTernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AppendCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryMatrixMatrixCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryMatrixScalarCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CentralMomentCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.IndexingCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMChainCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MMTSJCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantilePickCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuantileSortCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReorgCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ReshapeCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SpoofCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryFrameScalarCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryMatrixCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction.VariableOperationCode;
import org.apache.sysds.runtime.instructions.spark.AggregateBinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateTernarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AggregateUnarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGAlignedSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendGSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendMSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendRSPInstruction;
import org.apache.sysds.runtime.instructions.spark.AppendSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinaryMatrixMatrixSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinaryMatrixScalarSPInstruction;
import org.apache.sysds.runtime.instructions.spark.BinarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.CastSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CentralMomentSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CovarianceSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CpmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CtableSPInstruction;
import org.apache.sysds.runtime.instructions.spark.CumulativeOffsetSPInstruction;
import org.apache.sysds.runtime.instructions.spark.IndexingSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MapmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.MultiReturnParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ParameterizedBuiltinSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantilePickSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuantileSortSPInstruction;
import org.apache.sysds.runtime.instructions.spark.QuaternarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReblockSPInstruction;
import org.apache.sysds.runtime.instructions.spark.ReorgSPInstruction;
import org.apache.sysds.runtime.instructions.spark.RmmSPInstruction;
import org.apache.sysds.runtime.instructions.spark.SpoofSPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernaryFrameScalarSPInstruction;
import org.apache.sysds.runtime.instructions.spark.TernarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnaryMatrixSPInstruction;
import org.apache.sysds.runtime.instructions.spark.UnarySPInstruction;
import org.apache.sysds.runtime.instructions.spark.WriteSPInstruction;

public class FEDInstructionUtils {
	
	private static final String[] PARAM_BUILTINS = new String[]{
		"replace", "rmempty", "lowertri", "uppertri", "transformdecode", "transformapply", "tokenize"};

	public static boolean noFedRuntimeConversion = false;
	
	// private static final Log LOG = LogFactory.getLog(FEDInstructionUtils.class.getName());

	// This is currently a rather simplistic to our solution of replacing instructions with their correct federated
	// counterpart, since we do not propagate the information that a matrix is federated, therefore we can not decide
	// to choose a federated instruction earlier.

	/**
	 * Check and replace CP instructions with federated instructions if the instruction match criteria.
	 *
	 * @param inst The instruction to analyse
	 * @param ec The Execution Context
	 * @return The potentially modified instruction
	 */
	public static Instruction checkAndReplaceCP(Instruction inst, ExecutionContext ec) {
		if ( !noFedRuntimeConversion ){
			FEDInstruction fedinst = null;
			if (inst instanceof AggregateBinaryCPInstruction) {
				AggregateBinaryCPInstruction instruction = (AggregateBinaryCPInstruction) inst;
				if( instruction.input1.isMatrix() && instruction.input2.isMatrix()) {
					MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
					MatrixObject mo2 = ec.getMatrixObject(instruction.input2);
					if ( (mo1.isFederated(FType.ROW) && mo1.isFederatedExcept(FType.BROADCAST))
						|| (mo2.isFederated(FType.ROW) && mo2.isFederatedExcept(FType.BROADCAST))
						|| (mo1.isFederated(FType.COL) && mo1.isFederatedExcept(FType.BROADCAST))) {
						fedinst = AggregateBinaryFEDInstruction.parseInstruction(instruction);
					}
				}
			}
			else if( inst instanceof MMChainCPInstruction) {
				MMChainCPInstruction linst = (MMChainCPInstruction) inst;
				MatrixObject mo = ec.getMatrixObject(linst.input1);
				if( mo.isFederated(FType.ROW) )
					fedinst = MMChainFEDInstruction.parseInstruction(linst);
			}
			else if( inst instanceof MMTSJCPInstruction ) {
				MMTSJCPInstruction linst = (MMTSJCPInstruction) inst;
				MatrixObject mo = ec.getMatrixObject(linst.input1);
				if( (mo.isFederated(FType.ROW) && mo.isFederatedExcept(FType.BROADCAST) && linst.getMMTSJType().isLeft()) ||
					(mo.isFederated(FType.COL) && mo.isFederatedExcept(FType.BROADCAST) && linst.getMMTSJType().isRight()))
					fedinst = TsmmFEDInstruction.parseInstruction(linst);
			}
			else if (inst instanceof UnaryCPInstruction && ! (inst instanceof IndexingCPInstruction)) {
				UnaryCPInstruction instruction = (UnaryCPInstruction) inst;
				if(inst instanceof ReorgCPInstruction && (inst.getOpcode().equals("r'") || inst.getOpcode().equals("rdiag")
					|| inst.getOpcode().equals("rev"))) {
					ReorgCPInstruction rinst = (ReorgCPInstruction) inst;
					CacheableData<?> mo = ec.getCacheableData(rinst.input1);

					if((mo instanceof MatrixObject || mo instanceof FrameObject)
						&& mo.isFederatedExcept(FType.BROADCAST) )
						fedinst = ReorgFEDInstruction.parseInstruction(rinst);
				}
				else if(instruction.input1 != null && instruction.input1.isMatrix()
					&& ec.containsVariable(instruction.input1)) {

					MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
					if( mo1.isFederatedExcept(FType.BROADCAST) ) {
						if(instruction instanceof CentralMomentCPInstruction)
							fedinst = CentralMomentFEDInstruction.parseInstruction((CentralMomentCPInstruction) inst);
						else if(inst instanceof QuantileSortCPInstruction) {
							if(mo1.isFederated(FType.ROW) || mo1.getFedMapping().getFederatedRanges().length == 1 && mo1.isFederated(FType.COL))
								fedinst = QuantileSortFEDInstruction.parseInstruction((QuantileSortCPInstruction) inst);
						}
						else if(inst instanceof ReshapeCPInstruction)
							fedinst = ReshapeFEDInstruction.parseInstruction((ReshapeCPInstruction) inst);
						else if(inst instanceof AggregateUnaryCPInstruction &&
							((AggregateUnaryCPInstruction) instruction).getAUType() == AggregateUnaryCPInstruction.AUType.DEFAULT)
							fedinst = AggregateUnaryFEDInstruction.parseInstruction((AggregateUnaryCPInstruction) inst);
						else if(inst instanceof UnaryMatrixCPInstruction) {
							if(UnaryMatrixFEDInstruction.isValidOpcode(inst.getOpcode()) &&
								!(inst.getOpcode().equalsIgnoreCase("ucumk+*") && mo1.isFederated(FType.COL)))
								fedinst = UnaryMatrixFEDInstruction.parseInstruction((UnaryMatrixCPInstruction) inst);
						}
					}
				}
			}
			else if (inst instanceof BinaryCPInstruction) {
				BinaryCPInstruction instruction = (BinaryCPInstruction) inst;
				if((instruction.input1.isMatrix() &&
					ec.getMatrixObject(instruction.input1).isFederatedExcept(FType.BROADCAST)) ||
					(instruction.input2 != null && instruction.input2.isMatrix() &&
						ec.getMatrixObject(instruction.input2).isFederatedExcept(FType.BROADCAST))) {
					if(instruction instanceof AppendCPInstruction)
						fedinst = AppendFEDInstruction.parseInstruction((AppendCPInstruction) inst);
					else if(instruction instanceof QuantilePickCPInstruction)
						fedinst = QuantilePickFEDInstruction.parseInstruction((QuantilePickCPInstruction) inst);
					else if(instruction instanceof CovarianceCPInstruction &&
						(ec.getMatrixObject(instruction.input1).isFederated(FType.ROW) ||
							ec.getMatrixObject(instruction.input2).isFederated(FType.ROW)))
						fedinst = CovarianceFEDInstruction.parseInstruction((CovarianceCPInstruction) inst);
					else if(instruction instanceof BinaryMatrixMatrixCPInstruction)
						fedinst = BinaryMatrixMatrixFEDInstruction
							.parseInstruction((BinaryMatrixMatrixCPInstruction) inst);
					else if(instruction instanceof BinaryMatrixScalarCPInstruction)
						fedinst = BinaryMatrixScalarFEDInstruction
							.parseInstruction((BinaryMatrixScalarCPInstruction) inst);
				}
			}
			else if( inst instanceof ParameterizedBuiltinCPInstruction ) {
				ParameterizedBuiltinCPInstruction pinst = (ParameterizedBuiltinCPInstruction) inst;
				if( ArrayUtils.contains(PARAM_BUILTINS, pinst.getOpcode()) && pinst.getTarget(ec).isFederatedExcept(FType.BROADCAST) )
					fedinst = ParameterizedBuiltinFEDInstruction.parseInstruction(pinst);
			}
			else if (inst instanceof MultiReturnParameterizedBuiltinCPInstruction) {
				MultiReturnParameterizedBuiltinCPInstruction minst = (MultiReturnParameterizedBuiltinCPInstruction) inst;
				if(minst.getOpcode().equals("transformencode") && minst.input1.isFrame()) {
					CacheableData<?> fo = ec.getCacheableData(minst.input1);
					if(fo.isFederatedExcept(FType.BROADCAST)) {
						fedinst = MultiReturnParameterizedBuiltinFEDInstruction.parseInstruction(minst);
					}
				}
			}
			else if(inst instanceof IndexingCPInstruction) {
				// matrix and frame indexing
				IndexingCPInstruction minst = (IndexingCPInstruction) inst;
				if((minst.input1.isMatrix() || minst.input1.isFrame())
					&& ec.getCacheableData(minst.input1).isFederatedExcept(FType.BROADCAST)) {
					fedinst = IndexingFEDInstruction.parseInstruction(minst);
				}
			}
			else if(inst instanceof TernaryCPInstruction) {
				TernaryCPInstruction tinst = (TernaryCPInstruction) inst;
				if(inst.getOpcode().equals("_map") && inst instanceof TernaryFrameScalarCPInstruction && !inst.getInstructionString().contains("UtilFunctions")
					&& tinst.input1.isFrame() && ec.getFrameObject(tinst.input1).isFederated()) {
					long margin = ec.getScalarInput(tinst.input3).getLongValue();
					FrameObject fo = ec.getFrameObject(tinst.input1);
					if(margin == 0 || (fo.isFederated(FType.ROW) && margin == 1) || (fo.isFederated(FType.COL) && margin == 2))
						fedinst = TernaryFrameScalarFEDInstruction.parseInstruction((TernaryFrameScalarCPInstruction) inst);
				}
				else if((tinst.input1.isMatrix() && ec.getCacheableData(tinst.input1).isFederatedExcept(FType.BROADCAST))
					|| (tinst.input2.isMatrix() && ec.getCacheableData(tinst.input2).isFederatedExcept(FType.BROADCAST))
					|| (tinst.input3.isMatrix() && ec.getCacheableData(tinst.input3).isFederatedExcept(FType.BROADCAST))) {
					fedinst = TernaryFEDInstruction.parseInstruction(tinst);
				}
			}
			else if(inst instanceof VariableCPInstruction ){
				VariableCPInstruction ins = (VariableCPInstruction) inst;
				if(ins.getVariableOpcode() == VariableOperationCode.Write
					&& ins.getInput1().isMatrix()
					&& ins.getInput3().getName().contains("federated")){
					fedinst = VariableFEDInstruction.parseInstruction(ins);
				}
				else if(ins.getVariableOpcode() == VariableOperationCode.CastAsFrameVariable
					&& ins.getInput1().isMatrix()
					&& ec.getCacheableData(ins.getInput1()).isFederatedExcept(FType.BROADCAST)){
					fedinst = VariableFEDInstruction.parseInstruction(ins);
				}
				else if(ins.getVariableOpcode() == VariableOperationCode.CastAsMatrixVariable
					&& ins.getInput1().isFrame()
					&& ec.getCacheableData(ins.getInput1()).isFederatedExcept(FType.BROADCAST)){
					fedinst = VariableFEDInstruction.parseInstruction(ins);
				}
			}
			else if(inst instanceof AggregateTernaryCPInstruction){
				AggregateTernaryCPInstruction ins = (AggregateTernaryCPInstruction) inst;
				if(ins.input1.isMatrix() && ec.getCacheableData(ins.input1).isFederatedExcept(FType.BROADCAST)
					&& ins.input2.isMatrix() && ec.getCacheableData(ins.input2).isFederatedExcept(FType.BROADCAST)) {
					fedinst = AggregateTernaryFEDInstruction.parseInstruction(ins);
				}
			}
			else if(inst instanceof QuaternaryCPInstruction) {
				QuaternaryCPInstruction instruction = (QuaternaryCPInstruction) inst;
				Data data = ec.getVariable(instruction.input1);
				if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
					fedinst = QuaternaryFEDInstruction.parseInstruction(instruction);
			}
			else if(inst instanceof SpoofCPInstruction) {
				SpoofCPInstruction ins = (SpoofCPInstruction) inst;
				Class<?> scla = ins.getOperatorClass().getSuperclass();
				if(((scla == SpoofCellwise.class || scla == SpoofMultiAggregate.class || scla == SpoofOuterProduct.class)
					&& SpoofFEDInstruction.isFederated(ec, ins.getInputs(), scla))
					|| (scla == SpoofRowwise.class && SpoofFEDInstruction.isFederated(ec, FType.ROW, ins.getInputs(), scla))) {
					fedinst = SpoofFEDInstruction.parseInstruction(ins);
				}
			}
			else if(inst instanceof CtableCPInstruction) {
				CtableCPInstruction cinst = (CtableCPInstruction) inst;
				if((inst.getOpcode().equalsIgnoreCase("ctable") || inst.getOpcode().equalsIgnoreCase("ctableexpand"))
					&& ( ec.getCacheableData(cinst.input1).isFederated(FType.ROW)
					|| (cinst.input2.isMatrix() && ec.getCacheableData(cinst.input2).isFederated(FType.ROW))
					|| (cinst.input3.isMatrix() && ec.getCacheableData(cinst.input3).isFederated(FType.ROW))))
					fedinst = CtableFEDInstruction.parseInstruction(cinst);
			}

			//set thread id for federated context management
			if( fedinst != null ) {
				fedinst.setTID(ec.getTID());
				return fedinst;
			}
		}

		return inst;
	}

	public static Instruction checkAndReplaceSP(Instruction inst, ExecutionContext ec) {
		FEDInstruction fedinst = null;
		if(inst instanceof CastSPInstruction){
			CastSPInstruction ins = (CastSPInstruction) inst;
			if((ins.getOpcode().equalsIgnoreCase(OpOp1.CAST_AS_FRAME.toString())
					|| ins.getOpcode().equalsIgnoreCase(OpOp1.CAST_AS_MATRIX.toString()))
				&& ins.input1.isMatrix() && ec.getCacheableData(ins.input1).isFederatedExcept(FType.BROADCAST)){
				fedinst = CastFEDInstruction.parseInstruction(ins);
			}
		}
		else if (inst instanceof WriteSPInstruction) {
			WriteSPInstruction instruction = (WriteSPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if (data instanceof CacheableData && ((CacheableData<?>) data).isFederated()) {
				// Write spark instruction can not be executed for federated matrix objects (tries to get rdds which do
				// not exist), therefore we replace the instruction with the VariableCPInstruction.
				return VariableCPInstruction.parseInstruction(instruction.getInstructionString());
			}
		}
		else if(inst instanceof QuaternarySPInstruction) {
			QuaternarySPInstruction instruction = (QuaternarySPInstruction) inst;
			Data data = ec.getVariable(instruction.input1);
			if(data instanceof MatrixObject && ((MatrixObject) data).isFederated())
				fedinst = QuaternaryFEDInstruction.parseInstruction(instruction);
		}
		else if(inst instanceof SpoofSPInstruction) {
			SpoofSPInstruction ins = (SpoofSPInstruction) inst;
			Class<?> scla = ins.getOperatorClass().getSuperclass();
			if(((scla == SpoofCellwise.class || scla == SpoofMultiAggregate.class || scla == SpoofOuterProduct.class)
					&& SpoofFEDInstruction.isFederated(ec, ins.getInputs(), scla))
				|| (scla == SpoofRowwise.class && SpoofFEDInstruction.isFederated(ec, FType.ROW, ins.getInputs(), scla))) {
				fedinst = SpoofFEDInstruction.parseInstruction(ins);
			}
		}
		else if (inst instanceof UnarySPInstruction && ! (inst instanceof IndexingSPInstruction)) {
			UnarySPInstruction instruction = (UnarySPInstruction) inst;
			if (inst instanceof CentralMomentSPInstruction) {
				CentralMomentSPInstruction cinstruction = (CentralMomentSPInstruction) inst;
				Data data = ec.getVariable(cinstruction.input1);
				if (data instanceof MatrixObject && ((MatrixObject) data).isFederated() && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
					fedinst = CentralMomentFEDInstruction.parseInstruction(cinstruction);
			} else if (inst instanceof QuantileSortSPInstruction) {
				QuantileSortSPInstruction qinstruction = (QuantileSortSPInstruction) inst;
				Data data = ec.getVariable(qinstruction.input1);
				if (data instanceof MatrixObject && ((MatrixObject) data).isFederated() && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
					fedinst = QuantileSortFEDInstruction.parseInstruction(qinstruction);
			}
			else if (inst instanceof AggregateUnarySPInstruction) {
				AggregateUnarySPInstruction auinstruction = (AggregateUnarySPInstruction) inst;
				Data data = ec.getVariable(auinstruction.input1);
				if(data instanceof MatrixObject && ((MatrixObject) data).isFederated() && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
					if(ArrayUtils.contains(new String[]{"uarimin", "uarimax"}, auinstruction.getOpcode())) {
						if(((MatrixObject) data).getFedMapping().getType() == FType.ROW)
							fedinst = AggregateUnaryFEDInstruction.parseInstruction(auinstruction);
					}
					else
						fedinst = AggregateUnaryFEDInstruction.parseInstruction(auinstruction);
			}
			else if(inst instanceof ReorgSPInstruction && (inst.getOpcode().equals("r'") || inst.getOpcode().equals("rdiag")
				|| inst.getOpcode().equals("rev"))) {
				ReorgSPInstruction rinst = (ReorgSPInstruction) inst;
				CacheableData<?> mo = ec.getCacheableData(rinst.input1);
				if((mo instanceof MatrixObject || mo instanceof FrameObject) && mo.isFederated() && mo.isFederatedExcept(FType.BROADCAST))
					fedinst = ReorgFEDInstruction.parseInstruction(rinst);
			}
			else if(inst instanceof ReblockSPInstruction && instruction.input1 != null && (instruction.input1.isFrame() || instruction.input1.isMatrix())) {
				ReblockSPInstruction rinst = (ReblockSPInstruction)  instruction;
				CacheableData<?> data = ec.getCacheableData(rinst.input1);
				if(data.isFederatedExcept(FType.BROADCAST))
					fedinst = ReblockFEDInstruction.parseInstruction((ReblockSPInstruction) inst);
			}
			else if(instruction.input1 != null && instruction.input1.isMatrix() && ec.containsVariable(instruction.input1)) {
				MatrixObject mo1 = ec.getMatrixObject(instruction.input1);
				if(mo1.isFederatedExcept(FType.BROADCAST)) {
					if(instruction.getOpcode().equalsIgnoreCase("cm"))
						fedinst = CentralMomentFEDInstruction.parseInstruction((CentralMomentCPInstruction)inst);
					else if(inst.getOpcode().equalsIgnoreCase("qsort")) {
						if(mo1.getFedMapping().getFederatedRanges().length == 1)
							fedinst = QuantileSortFEDInstruction.parseInstruction(inst.getInstructionString(), false);
					}
					else if(inst.getOpcode().equalsIgnoreCase("rshape")) {
						fedinst = ReshapeFEDInstruction.parseInstruction(inst.getInstructionString());
					}
					else if(inst instanceof UnaryMatrixSPInstruction) {
						if(UnaryMatrixFEDInstruction.isValidOpcode(inst.getOpcode()))
							fedinst = UnaryMatrixFEDInstruction.parseInstruction((UnaryMatrixSPInstruction) inst);
					}
				}
			}
		}
		else if (inst instanceof BinarySPInstruction) {
			BinarySPInstruction instruction = (BinarySPInstruction) inst;
			if (inst instanceof MapmmSPInstruction || inst instanceof CpmmSPInstruction || inst instanceof RmmSPInstruction) {
				Data data = ec.getVariable(instruction.input1);
				if (data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST)) {
					fedinst = MMFEDInstruction.parseInstruction((AggregateBinarySPInstruction) instruction);
				}
			}
			else
				if(inst instanceof QuantilePickSPInstruction) {
				QuantilePickSPInstruction qinstruction = (QuantilePickSPInstruction) inst;
				Data data = ec.getVariable(qinstruction.input1);
				if(data instanceof MatrixObject && ((MatrixObject) data).isFederatedExcept(FType.BROADCAST))
					fedinst = QuantilePickFEDInstruction.parseInstruction(qinstruction);
			}
			else if (inst instanceof AppendGAlignedSPInstruction || inst instanceof AppendGSPInstruction
				|| inst instanceof AppendMSPInstruction || inst instanceof AppendRSPInstruction) {
				BinarySPInstruction ainstruction = (BinarySPInstruction) inst;
				Data data1 = ec.getVariable(ainstruction.input1);
				Data data2 = ec.getVariable(ainstruction.input2);
				if ((data1 instanceof MatrixObject && ((MatrixObject) data1).isFederatedExcept(FType.BROADCAST))
					|| (data2 instanceof MatrixObject && ((MatrixObject) data2).isFederatedExcept(FType.BROADCAST))) {
					fedinst = AppendFEDInstruction.parseInstruction((AppendSPInstruction) instruction);
				}
			}
			else if (inst instanceof BinaryMatrixScalarSPInstruction) {
				Data data = ec.getVariable(instruction.input1);
				if(data instanceof MatrixObject && ((MatrixObject)data).isFederatedExcept(FType.BROADCAST)) {
					fedinst = BinaryMatrixScalarFEDInstruction.parseInstruction((BinaryMatrixScalarSPInstruction) inst);
				}
			}
			else if (inst instanceof BinaryMatrixMatrixSPInstruction) {
				Data data = ec.getVariable(instruction.input1);
				if(data instanceof MatrixObject && ((MatrixObject)data).isFederatedExcept(FType.BROADCAST)) {
					fedinst = BinaryMatrixMatrixFEDInstruction.parseInstruction((BinaryMatrixMatrixSPInstruction) inst);
				}
			}
			else if( (instruction.input1.isMatrix() && ec.getCacheableData(instruction.input1).isFederatedExcept(FType.BROADCAST))
				|| (instruction.input2.isMatrix() && ec.getMatrixObject(instruction.input2).isFederatedExcept(FType.BROADCAST))) {
				if(inst instanceof CovarianceSPInstruction && (ec.getMatrixObject(instruction.input1)
					.isFederated(FType.ROW) || ec.getMatrixObject(instruction.input2).isFederated(FType.ROW)))
					fedinst = CovarianceFEDInstruction.parseInstruction((CovarianceSPInstruction) inst);
				else if(inst instanceof CumulativeOffsetSPInstruction) {
					fedinst = CumulativeOffsetFEDInstruction.parseInstruction((CumulativeOffsetSPInstruction) inst);
				}
				else
					fedinst = BinaryFEDInstruction.parseInstruction(InstructionUtils
						.concatOperands(inst.getInstructionString(), FEDInstruction.FederatedOutput.NONE.name()));
			}
		}
		else if( inst instanceof ParameterizedBuiltinSPInstruction) {
			ParameterizedBuiltinSPInstruction pinst = (ParameterizedBuiltinSPInstruction) inst;
			if( pinst.getOpcode().equalsIgnoreCase("replace") && pinst.getTarget(ec).isFederatedExcept(FType.BROADCAST) )
				fedinst = ParameterizedBuiltinFEDInstruction.parseInstruction(pinst);
		}
		else if (inst instanceof MultiReturnParameterizedBuiltinSPInstruction) {
			MultiReturnParameterizedBuiltinSPInstruction minst = (MultiReturnParameterizedBuiltinSPInstruction) inst;
			if(minst.getOpcode().equals("transformencode") && minst.input1.isFrame()) {
				CacheableData<?> fo = ec.getCacheableData(minst.input1);
				if(fo.isFederatedExcept(FType.BROADCAST)) {
					fedinst = MultiReturnParameterizedBuiltinFEDInstruction.parseInstruction(minst);
				}
			}
		}
		else if(inst instanceof IndexingSPInstruction) {
			// matrix and frame indexing
			IndexingSPInstruction minst = (IndexingSPInstruction) inst;
			if((minst.input1.isMatrix() || minst.input1.isFrame())
				&& ec.getCacheableData(minst.input1).isFederatedExcept(FType.BROADCAST)) {
				fedinst = IndexingFEDInstruction.parseInstruction(minst);
			}
		}
		else if(inst instanceof TernarySPInstruction) {
			TernarySPInstruction tinst = (TernarySPInstruction) inst;
			if(inst.getOpcode().equals("_map") && inst instanceof TernaryFrameScalarSPInstruction && !inst.getInstructionString().contains("UtilFunctions")
				&& tinst.input1.isFrame() && ec.getFrameObject(tinst.input1).isFederated()) {
				long margin = ec.getScalarInput(tinst.input3).getLongValue();
				FrameObject fo = ec.getFrameObject(tinst.input1);
				if(margin == 0 || (fo.isFederated(FType.ROW) && margin == 1) || (fo.isFederated(FType.COL) && margin == 2))
					fedinst = TernaryFrameScalarFEDInstruction.parseInstruction((TernaryFrameScalarSPInstruction) tinst);
			} else if((tinst.input1.isMatrix() && ec.getCacheableData(tinst.input1).isFederatedExcept(FType.BROADCAST))
				|| (tinst.input2.isMatrix() && ec.getCacheableData(tinst.input2).isFederatedExcept(FType.BROADCAST))
				|| (tinst.input3.isMatrix() && ec.getCacheableData(tinst.input3).isFederatedExcept(FType.BROADCAST))) {
				fedinst = TernaryFEDInstruction.parseInstruction(tinst);
			}
		}
		else if(inst instanceof AggregateTernarySPInstruction){
			AggregateTernarySPInstruction ins = (AggregateTernarySPInstruction) inst;
			if(ins.input1.isMatrix() && ec.getCacheableData(ins.input1).isFederatedExcept(FType.BROADCAST) && ins.input2.isMatrix() &&
				ec.getCacheableData(ins.input2).isFederatedExcept(FType.BROADCAST)) {
				fedinst = AggregateTernaryFEDInstruction.parseInstruction(ins);
			}
		}
		else if(inst instanceof CtableSPInstruction) {
			CtableSPInstruction cinst = (CtableSPInstruction) inst;
			if(inst.getOpcode().equalsIgnoreCase("ctable")
				&& ( ec.getCacheableData(cinst.input1).isFederated(FType.ROW)
				|| (cinst.input2.isMatrix() && ec.getCacheableData(cinst.input2).isFederated(FType.ROW))
				|| (cinst.input3.isMatrix() && ec.getCacheableData(cinst.input3).isFederated(FType.ROW))))
				fedinst = CtableFEDInstruction.parseInstruction(cinst);
		}

		//set thread id for federated context management
		if( fedinst != null ) {
			fedinst.setTID(ec.getTID());
			return fedinst;
		}

		return inst;
	}
}
