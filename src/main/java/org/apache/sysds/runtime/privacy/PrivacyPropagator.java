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

package org.apache.sysds.runtime.privacy;

import java.util.*;

import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BuiltinNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.ComputationCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CovarianceCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.MultiReturnBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.SqlCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.finegrained.DataRange;
import org.apache.sysds.runtime.privacy.finegrained.FineGrainedPrivacy;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

/**
 * Class with static methods merging privacy constraints of operands 
 * in expressions to generate the privacy constraints of the output. 
 */
public class PrivacyPropagator
{
	public static Data parseAndSetPrivacyConstraint(Data cd, JSONObject mtd)
		throws JSONException
	{
		if ( mtd.containsKey(DataExpression.PRIVACY) ) {
			String privacyLevel = mtd.getString(DataExpression.PRIVACY);
			if ( privacyLevel != null )
				cd.setPrivacyConstraints(new PrivacyConstraint(PrivacyLevel.valueOf(privacyLevel)));
		}
		return cd;
	}
	
	public static PrivacyConstraint mergeBinary(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2) {
		if (privacyConstraint1 != null && privacyConstraint2 != null){
			PrivacyLevel privacyLevel1 = privacyConstraint1.getPrivacyLevel();
			PrivacyLevel privacyLevel2 = privacyConstraint2.getPrivacyLevel();

			// One of the inputs are private, hence the output must be private.
			if (privacyLevel1 == PrivacyLevel.Private || privacyLevel2 == PrivacyLevel.Private)
				return new PrivacyConstraint(PrivacyLevel.Private);
			// One of the inputs are private with aggregation allowed, but none of the inputs are completely private,
			// hence the output must be private with aggregation.
			else if (privacyLevel1 == PrivacyLevel.PrivateAggregation || privacyLevel2 == PrivacyLevel.PrivateAggregation)
				return new PrivacyConstraint(PrivacyLevel.PrivateAggregation);
			// Both inputs have privacy level "None", hence the privacy constraint can be removed.
			else  
				return null;
		}
		else if (privacyConstraint1 != null)
			return privacyConstraint1;
		else if (privacyConstraint2 != null)
			return privacyConstraint2;
		return null;
	}

	public static PrivacyConstraint mergeNary(PrivacyConstraint[] privacyConstraints){
		PrivacyConstraint mergedPrivacyConstraint = privacyConstraints[0];
		for ( int i = 1; i < privacyConstraints.length; i++ ){
			mergedPrivacyConstraint = mergeBinary(mergedPrivacyConstraint, privacyConstraints[i]);
		}
		return mergedPrivacyConstraint;
	}

	public static Instruction preprocessInstruction(Instruction inst, ExecutionContext ec){
		switch ( inst.getType() ){
			case CONTROL_PROGRAM:
				return preprocessCPInstructionFineGrained( (CPInstruction) inst, ec );
			case BREAKPOINT:
			case SPARK:
			case GPU:
			case FEDERATED:
				return inst;
			default:
				throwExceptionIfPrivacyActivated(inst);
				return inst;
		}
	}

	public static Instruction preprocessCPInstructionFineGrained(CPInstruction inst, ExecutionContext ec){
		switch ( inst.getCPInstructionType() ){
			case AggregateBinary:
				if ( inst instanceof AggregateBinaryCPInstruction ){
					// This can only be a matrix multiplication and it does not count as an aggregation in terms of privacy.
					return preprocessAggregateBinaryCPInstruction((AggregateBinaryCPInstruction)inst, ec);
				} else if ( inst instanceof CovarianceCPInstruction ){
					return preprocessCovarianceCPInstruction((CovarianceCPInstruction)inst, ec);
				} else preprocessInstructionSimple(inst, ec);
			case AggregateTernary:
				//TODO: Support propagation of fine-grained privacy constraints
				return preprocessTernaryCPInstruction((ComputationCPInstruction) inst, ec);
			case AggregateUnary:
				// Assumption: aggregates in one or several dimensions, number of dimensions may change, only certain slices of the data may be aggregated upon, elements do not change position
				return preprocessAggregateUnaryCPInstruction((AggregateUnaryCPInstruction)inst, ec);
			case Append:
			case Binary:
				// TODO: Support propagation of fine-grained privacy constraints
				return preprocessBinaryCPInstruction((BinaryCPInstruction) inst, ec);
			case Builtin:
			case BuiltinNary:
				//TODO: Support propagation of fine-grained privacy constraints
				return preprocessBuiltinNary((BuiltinNaryCPInstruction) inst, ec);
			/*case CentralMoment:
				break;
			case Compression:
				break;
			case Covariance:
				break;
			case Ctable:
				break;
			case Dnn:
				break;
			 */
			case FCall:
				//TODO: Support propagation of fine-grained privacy constraints
				return preprocessExternal((FunctionCallCPInstruction) inst, ec);
			/*
			case MMChain:
				break;
			case MMTSJ:
				break;
			case MatrixIndexing:
				break;*/
			case MultiReturnBuiltin:
			case MultiReturnParameterizedBuiltin:
				// TODO: Support propagation of fine-grained privacy constraints
				return preprocessMultiReturn((ComputationCPInstruction)inst, ec);
			/*case PMMJ:
				break;*/
			case ParameterizedBuiltin:
				// TODO: Support propagation of fine-grained privacy constraints
				return preprocessParameterizedBuiltin((ParameterizedBuiltinCPInstruction) inst, ec);
			/*case Partition:
				break;
			case QPick:
				break;
			case QSort:
				break;*/
			case Quaternary:
				// TODO: Support propagation of fine-grained privacy constraints
				return preprocessQuaternary((QuaternaryCPInstruction) inst, ec);
			/*case Rand:
				break;*/
			case Reorg:
				// TODO: Support propagation of fine-grained privacy constraints
				return preprocessUnaryCPInstruction((UnaryCPInstruction) inst, ec);
			/*case Reshape:
				break;
			case SpoofFused:
				break;
			case Sql:
				break;
			case StringInit:
				break;*/
			case Ternary:
				// TODO: Support propagation of fine-grained privacy constraints
				return preprocessTernaryCPInstruction((ComputationCPInstruction) inst, ec);
			/*case UaggOuterChain:
				break;*/
			case Unary:
				// Assumption: No aggregation, elements do not change position, no change of dimensions
				return preprocessUnaryCPInstruction((UnaryCPInstruction) inst, ec);
			case Variable:
				return preprocessVariableCPInstruction((VariableCPInstruction) inst, ec);
			default:
				return preprocessInstructionSimple(inst, ec);
			
		}
	}

	/**
	 * Throw exception if privacy constraint activated for instruction or for input to instruction.
	 * @param inst covariance instruction
	 * @param ec execution context
	 * @return input instruction if privacy constraints are not activated
	 */
	private static Instruction preprocessCovarianceCPInstruction(CovarianceCPInstruction inst, ExecutionContext ec){
		throwExceptionIfPrivacyActivated(inst);
		for ( CPOperand input : inst.getInputs() ){
			PrivacyConstraint privacyConstraint = getInputPrivacyConstraint(ec, input);
			if ( privacyConstraint != null){
				throw new DMLPrivacyException("Input of instruction " + inst + " has privacy constraints activated, but the constraints are not propagated during preprocessing of instruction.");
			}
		}
		return inst;
	}

	private static Instruction preprocessAggregateBinaryCPInstruction(AggregateBinaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint privacyConstraint1 = getInputPrivacyConstraint(ec, inst.input1);
		PrivacyConstraint privacyConstraint2 = getInputPrivacyConstraint(ec, inst.input2);
		if ( (privacyConstraint1 != null && privacyConstraint1.hasConstraints()) 
			|| (privacyConstraint2 != null && privacyConstraint2.hasConstraints()) ){
				PrivacyConstraint mergedPrivacyConstraint;
				if ( (privacyConstraint1 != null && privacyConstraint1.hasFineGrainedConstraints() ) || (privacyConstraint2 != null && privacyConstraint2.hasFineGrainedConstraints() )){
					MatrixBlock input1 = ec.getMatrixInput(inst.input1.getName());
					MatrixBlock input2 = ec.getMatrixInput(inst.input2.getName());
					mergedPrivacyConstraint = matrixMultiplicationPropagation(input1, privacyConstraint1, input2, privacyConstraint2);
				}
				else {
					mergedPrivacyConstraint = mergeBinary(privacyConstraint1, privacyConstraint2);
					inst.setPrivacyConstraint(mergedPrivacyConstraint);
				}
				inst.output.setPrivacyConstraint(mergedPrivacyConstraint);
		}
		return inst;
	}

	public static Instruction preprocessBinaryCPInstruction(BinaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint privacyConstraint1 = getInputPrivacyConstraint(ec, inst.input1);
		PrivacyConstraint privacyConstraint2 = getInputPrivacyConstraint(ec, inst.input2);
		if ( privacyConstraint1 != null || privacyConstraint2 != null) {
			PrivacyConstraint mergedPrivacyConstraint = mergeBinary(privacyConstraint1, privacyConstraint2);
			inst.setPrivacyConstraint(mergedPrivacyConstraint);
			inst.output.setPrivacyConstraint(mergedPrivacyConstraint);
		}
		return inst;
	}

	/**
	 * Return the merged fine-grained privacy constraint of a matrix multiplication with the given privacy constraints.
	 * The current implementation has a tendency to create small ranges of privacy level private. These ranges could be merged
	 * to create fewer ranges spanning the same elements.
	 * @param input1 first input matrix block
	 * @param privacyConstraint1 privacy constraint of the first matrix
	 * @param input2 second input matrix block
	 * @param privacyConstraint2 privacy constraint of the second matrix
	 * @return merged privacy constraint
	 */
	public static PrivacyConstraint matrixMultiplicationPropagation(MatrixBlock input1, PrivacyConstraint privacyConstraint1, MatrixBlock input2, PrivacyConstraint privacyConstraint2){
		// If the overall privacy level is private, then the fine-grained constraints do not have to be checked.
		if ( (privacyConstraint1 != null && privacyConstraint1.getPrivacyLevel() == PrivacyLevel.Private) || (privacyConstraint2 != null && privacyConstraint2.getPrivacyLevel() == PrivacyLevel.Private) )
			return new PrivacyConstraint(PrivacyLevel.Private);
		
		boolean hasOverallConstraintAggregate = ( (privacyConstraint1 != null && privacyConstraint1.getPrivacyLevel() == PrivacyLevel.PrivateAggregation ) || ( privacyConstraint2 != null && privacyConstraint2.getPrivacyLevel() == PrivacyLevel.PrivateAggregation));
		PrivacyConstraint mergedConstraint = new PrivacyConstraint();
		if ( hasOverallConstraintAggregate )
			mergedConstraint.setPrivacyLevel(PrivacyLevel.PrivateAggregation);

		int r1 = input1.getNumRows();
		int c1 = input1.getNumColumns();
		int r2 = input2.getNumRows();
		int c2 = input2.getNumColumns();
		FineGrainedPrivacy mergedFineGrainedConstraints = mergedConstraint.getFineGrainedPrivacy();

		for (int i = 0; i < r1; i++){

			// Get privacy of first matrix row
			long[] beginRange1 = new long[]{i,0};
			long[] endRange1 = new long[]{i,c1};
			Map<DataRange, PrivacyLevel> privacyInRow = (privacyConstraint1 != null) ? privacyConstraint1.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(beginRange1, endRange1)) : new HashMap<>();
			
			for (int j = 0; j < c2; j++){
				// Get Privacy of Second matrix col
				long[] beginRange2 = new long[]{0,j};
				long[] endRange2 = new long[]{r2,j};
				Map<DataRange, PrivacyLevel> privacyInCol = (privacyConstraint2 != null) ? privacyConstraint2.getFineGrainedPrivacy().getPrivacyLevel(new DataRange(beginRange2, endRange2)) : new HashMap<>();
			
				// if any elements in the row or col has privacy level private or privateaggregate, 
				// then output element in the index should be same level.
				long[] beginRangeMerged = new long[]{i,j};
				long[] endRangeMerged = new long[]{i,j};
				if ( privacyInRow.containsValue(PrivacyLevel.Private) || privacyInCol.containsValue(PrivacyLevel.Private))
					mergedFineGrainedConstraints.put(new DataRange(beginRangeMerged, endRangeMerged), PrivacyLevel.Private);
				else if ( !hasOverallConstraintAggregate && (privacyInRow.containsValue(PrivacyLevel.PrivateAggregation) || privacyInCol.containsValue(PrivacyLevel.PrivateAggregation) ))
					mergedFineGrainedConstraints.put(new DataRange(beginRangeMerged, endRangeMerged), PrivacyLevel.PrivateAggregation);
			}
		}
		return mergedConstraint;
	}

	/**
	 * Propagate privacy constraint to output if any of the elements are private.
	 * Privacy constraint is always propagated to instruction.
	 * @param inst aggregate instruction
	 * @param ec execution context
	 * @return updated instruction with propagated privacy constraints
	 */
	private static Instruction preprocessAggregateUnaryCPInstruction(AggregateUnaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint privacyConstraint = getInputPrivacyConstraint(ec, inst.input1);
		if ( privacyConstraint != null ) {
			inst.setPrivacyConstraint(privacyConstraint);
			if ( inst.output != null){
				//Only propagate to output if any of the elements are private. 
				//It is an aggregation, hence the constraint can be removed in case of any other privacy level.
				if(privacyConstraint.hasPrivateElements())
					inst.output.setPrivacyConstraint(new PrivacyConstraint(PrivacyLevel.Private));
			}
		}
		return inst;
	}

	/**
	 * Throw exception if privacy constraints are activated or return instruction if privacy is not activated
	 * @param inst instruction
	 * @param ec execution context
	 * @return instruction
	 */
	public static Instruction preprocessInstructionSimple(Instruction inst, ExecutionContext ec){
		throwExceptionIfPrivacyActivated(inst);
		return inst;
	}


	public static Instruction preprocessExternal(FunctionCallCPInstruction inst, ExecutionContext ec){
		return mergePrivacyConstraintsFromInput(
			inst, 
			ec, 
			inst.getInputs(), 
			inst.getBoundOutputParamNames().toArray(new String[0])
		);
	}

	public static Instruction preprocessMultiReturn(ComputationCPInstruction inst, ExecutionContext ec){
		List<CPOperand> outputs = getOutputOperands(inst);
		return mergePrivacyConstraintsFromInput(inst, ec, inst.getInputs(), outputs);
	}

	public static Instruction preprocessParameterizedBuiltin(ParameterizedBuiltinCPInstruction inst, ExecutionContext ec){
		return mergePrivacyConstraintsFromInput(inst, ec, inst.getInputs(), inst.getOutput() );
	}

	private static Instruction mergePrivacyConstraintsFromInput(Instruction inst, ExecutionContext ec, CPOperand[] inputs, String[] outputNames){
		if ( inputs != null && inputs.length > 0 ){
			PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inputs);
			if ( privacyConstraints != null ){
				PrivacyConstraint mergedPrivacyConstraint = mergeNary(privacyConstraints);
				inst.setPrivacyConstraint(mergedPrivacyConstraint);
				if ( outputNames != null ){
					for (String outputName : outputNames)
						setOutputPrivacyConstraint(ec, mergedPrivacyConstraint, outputName);
				}
			}
		}
		return inst;
	}

	private static Instruction mergePrivacyConstraintsFromInput(Instruction inst, ExecutionContext ec, CPOperand[] inputs, CPOperand output){
		return mergePrivacyConstraintsFromInput(inst, ec, inputs, getSingletonList(output));
	}

	private static Instruction mergePrivacyConstraintsFromInput(Instruction inst, ExecutionContext ec, CPOperand[] inputs, List<CPOperand> outputs){
		if ( inputs != null && inputs.length > 0 ){
			PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inputs);
			if ( privacyConstraints != null ){
				PrivacyConstraint mergedPrivacyConstraint = mergeNary(privacyConstraints);
				inst.setPrivacyConstraint(mergedPrivacyConstraint);
				for ( CPOperand output : outputs ){
					if ( output != null ) {
						output.setPrivacyConstraint(mergedPrivacyConstraint);
					}
				}
			}
		}
		return inst;
	}

	public static Instruction preprocessBuiltinNary(BuiltinNaryCPInstruction inst, ExecutionContext ec){
		return mergePrivacyConstraintsFromInput(inst, ec, inst.getInputs(), inst.getOutput() );
	}

	public static Instruction preprocessQuaternary(QuaternaryCPInstruction inst, ExecutionContext ec){
		return mergePrivacyConstraintsFromInput(
			inst, 
			ec, 
			new CPOperand[] {inst.input1,inst.input2,inst.input3,inst.getInput4()},
			inst.output
		);
	}

	public static Instruction preprocessTernaryCPInstruction(ComputationCPInstruction inst, ExecutionContext ec){
		return mergePrivacyConstraintsFromInput(inst, ec, inst.getInputs(), inst.output);
	}

	public static Instruction preprocessUnaryCPInstruction(UnaryCPInstruction inst, ExecutionContext ec){
		return propagateInputPrivacy(inst, ec, inst.input1, inst.output);
	}

	public static Instruction preprocessVariableCPInstruction(VariableCPInstruction inst, ExecutionContext ec){
		switch ( inst.getVariableOpcode() ) {
			case CreateVariable:
				return propagateSecondInputPrivacy(inst, ec);
			case AssignVariable:
				return propagateInputPrivacy(inst, ec, inst.getInput1(), inst.getInput2());
			case CopyVariable:
			case MoveVariable:
			case RemoveVariableAndFile:
			case CastAsMatrixVariable:
			case CastAsFrameVariable:
			case Write:
			case SetFileName:
				return propagateFirstInputPrivacy(inst, ec);
			case RemoveVariable:
				return propagateAllInputPrivacy(inst, ec);
			case CastAsScalarVariable:
			case CastAsDoubleVariable:
			case CastAsIntegerVariable:
			case CastAsBooleanVariable:
				return propagateCastAsScalarVariablePrivacy(inst, ec);
			case Read:
				return inst;
			default:
				throwExceptionIfPrivacyActivated(inst);
				return inst;
		}
	}

	private static void throwExceptionIfPrivacyActivated(Instruction inst){
		if ( inst.getPrivacyConstraint() != null && inst.getPrivacyConstraint().hasConstraints() ) {
			throw new DMLPrivacyException("Instruction " + inst + " has privacy constraints activated, but the constraints are not propagated during preprocessing of instruction.");
		}
	}

	/**
	 * Propagate privacy from first input.
	 * @param inst Instruction
	 * @param ec execution context
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateCastAsScalarVariablePrivacy(VariableCPInstruction inst, ExecutionContext ec){
		inst = (VariableCPInstruction) propagateFirstInputPrivacy(inst, ec); 
		return inst;
	}

	/**
	 * Propagate privacy constraints from all inputs if privacy constraints are set.
	 * @param inst instruction
	 * @param ec execution context
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateAllInputPrivacy(VariableCPInstruction inst, ExecutionContext ec){
		return mergePrivacyConstraintsFromInput(
			inst, ec, inst.getInputs().toArray(new CPOperand[0]), inst.getOutput());
	}

	/**
	 * Propagate privacy constraint to instruction and output of instruction
	 * if data of first input is CacheableData and 
	 * privacy constraint is activated.
	 * @param inst VariableCPInstruction
	 * @param ec execution context
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateFirstInputPrivacy(VariableCPInstruction inst, ExecutionContext ec){
		return propagateInputPrivacy(inst, ec, inst.getInput1(), inst.getOutput());
	}

	/**
	 * Propagate privacy constraint to instruction and output of instruction
	 * if data of second input is CacheableData and 
	 * privacy constraint is activated.
	 * @param inst VariableCPInstruction
	 * @param ec execution context
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateSecondInputPrivacy(VariableCPInstruction inst, ExecutionContext ec){
		return propagateInputPrivacy(inst, ec, inst.getInput2(), inst.getOutput());
	}

	/**
	 * Propagate privacy constraint to instruction and output of instruction
	 * if data of the specified variable is CacheableData 
	 * and privacy constraint is activated
	 * @param inst instruction
	 * @param ec execution context
	 * @param inputOperand input from which the privacy constraint is found
	 * @param outputOperand output which the privacy constraint is propagated to
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateInputPrivacy(Instruction inst, ExecutionContext ec, CPOperand inputOperand, CPOperand outputOperand){
		PrivacyConstraint privacyConstraint = getInputPrivacyConstraint(ec, inputOperand);
		if ( privacyConstraint != null ) {
			inst.setPrivacyConstraint(privacyConstraint);
			if ( outputOperand != null)
				outputOperand.setPrivacyConstraint(privacyConstraint);
		}
		return inst;
	}

	/**
	 * Get privacy constraint of input data variable from execution context.
	 * @param ec execution context from which the data variable is retrieved
	 * @param input data variable from which the privacy constraint is retrieved
	 * @return privacy constraint of variable or null if privacy constraint is not set
	 */
	private static PrivacyConstraint getInputPrivacyConstraint(ExecutionContext ec, CPOperand input){
		if ( input != null && input.getName() != null){
			Data dd = ec.getVariable(input.getName());
			if ( dd != null )
				return dd.getPrivacyConstraint();
		}
		return null;
	}


	private static PrivacyConstraint[] getInputPrivacyConstraints(ExecutionContext ec, CPOperand[] inputs){
		if ( inputs != null && inputs.length > 0){
			boolean privacyFound = false;
			PrivacyConstraint[] privacyConstraints = new PrivacyConstraint[inputs.length];
			for ( int i = 0; i < inputs.length; i++ ){
				privacyConstraints[i] = getInputPrivacyConstraint(ec, inputs[i]);
				if ( privacyConstraints[i] != null )
					privacyFound = true;
			}
			if ( privacyFound )
				return privacyConstraints;
		}
		return null;
	}

	/**
	 * Set privacy constraint of data variable with outputName 
	 * if the variable exists and the privacy constraint is not null.
	 * @param ec execution context from which the data variable is retrieved
	 * @param privacyConstraint privacy constraint which the variable should have
	 * @param outputName name of variable that is retrieved from the execution context
	 */
	private static void setOutputPrivacyConstraint(ExecutionContext ec, PrivacyConstraint privacyConstraint, String outputName){
		if ( privacyConstraint != null ){
			Data dd = ec.getVariable(outputName);
			if ( dd != null ){
				dd.setPrivacyConstraints(privacyConstraint);
				ec.setVariable(outputName, dd);
			}
		}
	}

	public static void postProcessInstruction(Instruction inst, ExecutionContext ec){
		// if inst has output
		List<CPOperand> instOutputs = getOutputOperands(inst);
		if (!instOutputs.isEmpty()){
			for ( CPOperand output : instOutputs ){
				PrivacyConstraint outputPrivacyConstraint = output.getPrivacyConstraint();
				if ( privacyConstraintActivated(outputPrivacyConstraint) )
					setOutputPrivacyConstraint(ec, outputPrivacyConstraint, output.getName());
			}
		}
	}

	private static boolean privacyConstraintActivated(PrivacyConstraint instructionPrivacyConstraint){
		return instructionPrivacyConstraint != null && 
			(instructionPrivacyConstraint.privacyLevel == PrivacyLevel.Private 
			|| instructionPrivacyConstraint.privacyLevel == PrivacyLevel.PrivateAggregation);
	}

	@SuppressWarnings("unused")
	private static String[] getOutputVariableName(Instruction inst){
		String[] instructionOutputNames = null;
		// The order of the following statements is important
		if ( inst instanceof MultiReturnParameterizedBuiltinCPInstruction )
			instructionOutputNames = ((MultiReturnParameterizedBuiltinCPInstruction) inst).getOutputNames();
		else if ( inst instanceof MultiReturnBuiltinCPInstruction )
			instructionOutputNames = ((MultiReturnBuiltinCPInstruction) inst).getOutputNames();
		else if ( inst instanceof ComputationCPInstruction )
			instructionOutputNames = new String[]{((ComputationCPInstruction) inst).getOutputVariableName()};
		else if ( inst instanceof VariableCPInstruction )
			instructionOutputNames = new String[]{((VariableCPInstruction) inst).getOutputVariableName()};
		else if ( inst instanceof SqlCPInstruction )
			instructionOutputNames = new String[]{((SqlCPInstruction) inst).getOutputVariableName()};
		return instructionOutputNames;
	}

	private static List<CPOperand> getOutputOperands(Instruction inst){
		// The order of the following statements is important
		if ( inst instanceof MultiReturnParameterizedBuiltinCPInstruction )
			return ((MultiReturnParameterizedBuiltinCPInstruction) inst).getOutputs();
		else if ( inst instanceof MultiReturnBuiltinCPInstruction )
			return ((MultiReturnBuiltinCPInstruction) inst).getOutputs();
		else if ( inst instanceof ComputationCPInstruction )
			return getSingletonList(((ComputationCPInstruction) inst).getOutput());
		else if ( inst instanceof VariableCPInstruction )
			return getSingletonList(((VariableCPInstruction) inst).getOutput());
		else if ( inst instanceof SqlCPInstruction )
			return getSingletonList(((SqlCPInstruction) inst).getOutput());
		return new ArrayList<>();
	}

	private static List<CPOperand> getSingletonList(CPOperand operand){
		if ( operand != null)
			return new ArrayList<>(Collections.singletonList(operand));
		return new ArrayList<>();
	}
}
