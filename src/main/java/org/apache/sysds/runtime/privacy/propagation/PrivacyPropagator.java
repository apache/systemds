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

package org.apache.sysds.runtime.privacy.propagation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.privacy.DMLPrivacyException;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.runtime.privacy.PrivacyUtils;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

/**
 * Class with static methods merging privacy constraints of operands 
 * in expressions to generate the privacy constraints of the output. 
 */
public class PrivacyPropagator
{
	/**
	 * Parses the privacy constraint of the given metadata object
	 * and sets the field of the given Data if the privacy constraint is not null.
	 * @param cd data for which privacy constraint is set
	 * @param mtd metadata object
	 * @return data object with privacy constraint set
	 * @throws JSONException during parsing of metadata
	 */
	public static Data parseAndSetPrivacyConstraint(Data cd, JSONObject mtd)
		throws JSONException
	{
		PrivacyConstraint mtdPrivConstraint = parseAndReturnPrivacyConstraint(mtd);
		if ( mtdPrivConstraint != null )
			cd.setPrivacyConstraints(mtdPrivConstraint);
		return cd;
	}

	/**
	 * Parses the privacy constraint of the given metadata object
	 * or returns null if no privacy constraint is set in the metadata.
	 * @param mtd metadata
	 * @return privacy constraint parsed from metadata object
	 * @throws JSONException during parsing of metadata
	 */
	public static PrivacyConstraint parseAndReturnPrivacyConstraint(JSONObject mtd)
		throws JSONException
	{
		if ( mtd.containsKey(DataExpression.PRIVACY) ) {
			String privacyLevel = mtd.getString(DataExpression.PRIVACY);
			if ( privacyLevel != null )
				return new PrivacyConstraint(PrivacyLevel.valueOf(privacyLevel));
		}
		return null;
	}

	private static boolean anyInputHasLevel(PrivacyLevel[] inputLevels, PrivacyLevel targetLevel){
		return Arrays.stream(inputLevels).anyMatch(i -> i == targetLevel);
	}

	/**
	 * Returns the output privacy level based on the given input privacy levels and operator type.
	 * It represents the basic logic of privacy propagation:
	 *
	 * Unary input:
	 * Input   | NonAggregate | Aggregate
	 * -----------------------------------
	 * priv    | priv         | priv
	 * privAgg | privAgg      | none
	 * none    | none         | none
	 *
	 * Binary input:
	 * Input   			| NonAggregate 	| Aggregate
	 * --------------------------------------------
	 * priv-priv 		| priv 			| priv
	 * priv-privAgg 	| priv 			| priv
	 * priv-none 		| priv 			| priv
	 * privAgg-priv 	| priv 			| priv
	 * none-priv 		| priv 			| priv
	 * privAgg-privAgg 	| privAgg 		| none
	 * none-none 		| none 			| none
	 * privAgg-none 	| privAgg 		| none
	 * none-privAgg 	| privAgg 		| none
	 *
	 * @param inputLevels privacy levels of the input
	 * @param operatorType type of the operator which is either an aggregation (Aggregate) or not an aggregation (NonAggregate)
	 * @return output privacy level
	 */
	public static PrivacyLevel corePropagation(PrivacyLevel[] inputLevels, OperatorType operatorType){
		if (anyInputHasLevel(inputLevels, PrivacyLevel.Private))
			return PrivacyLevel.Private;
		if (operatorType == OperatorType.Aggregate)
			return PrivacyLevel.None;
		if (operatorType == OperatorType.NonAggregate && anyInputHasLevel(inputLevels,PrivacyLevel.PrivateAggregation))
			return PrivacyLevel.PrivateAggregation;
		return PrivacyLevel.None;
	}

	/**
	 * Merges the given privacy constraints with the core propagation using the given operator type.
	 * @param privacyConstraints array of privacy constraints to merge
	 * @param operatorType type of operation to use when merging with the core propagation
	 * @return merged privacy constraint
	 */
	private static PrivacyConstraint mergeNary(PrivacyConstraint[] privacyConstraints, OperatorType operatorType){
		PrivacyLevel[] privacyLevels = Arrays.stream(privacyConstraints)
			.map(constraint -> {
				if (constraint != null)
					return constraint.getPrivacyLevel();
				else return PrivacyLevel.None;
			})
			.toArray(PrivacyLevel[]::new);
		PrivacyLevel outputPrivacyLevel = corePropagation(privacyLevels, operatorType);
		return new PrivacyConstraint(outputPrivacyLevel);
	}

	/**
	 * Merges the input privacy constraints using the core propagation with NonAggregate operator type.
	 * @param privacyConstraint1 first privacy constraint
	 * @param privacyConstraint2 second privacy constraint
	 * @return merged privacy constraint
	 */
	public static PrivacyConstraint mergeBinary(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2) {
		if (privacyConstraint1 != null && privacyConstraint2 != null){
			PrivacyLevel[] privacyLevels = new PrivacyLevel[]{
				privacyConstraint1.getPrivacyLevel(),privacyConstraint2.getPrivacyLevel()};
			return new PrivacyConstraint(corePropagation(privacyLevels, OperatorType.NonAggregate));
		}
		else if (privacyConstraint1 != null)
			return privacyConstraint1;
		else if (privacyConstraint2 != null)
			return privacyConstraint2;
		return null;
	}

	/**
	 * Propagate privacy constraints from input hops to given hop.
	 * @param hop which the privacy constraints are propagated to
	 */
	public static void hopPropagation(Hop hop){
		hopPropagation(hop, hop.getInput());
	}

	/**
	 * Propagate privacy constraints from input hops to given hop.
	 * @param hop which the privacy constraints are propagated to
	 * @param inputHops inputs to given hop
	 */
	public static void hopPropagation(Hop hop, ArrayList<Hop> inputHops){
		PrivacyConstraint[] inputConstraints = inputHops.stream()
			.map(Hop::getPrivacy).toArray(PrivacyConstraint[]::new);
		OperatorType opType = getOpType(hop);
		hop.setPrivacy(mergeNary(inputConstraints, opType));
		if (opType == null && Arrays.stream(inputConstraints).anyMatch(Objects::nonNull))
			throw new DMLException("Input has constraint but hop type not recognized by PrivacyPropagator. " +
				"Hop is " + hop + " " + hop.getClass());
	}

	/**
	 * Get operator type of given hop.
	 * Returns null if hop type is not known.
	 * @param hop for which operator type is returned
	 * @return operator type of hop or null if hop type is unknown
	 */
	private static OperatorType getOpType(Hop hop){
		if ( hop instanceof TernaryOp || hop instanceof BinaryOp || hop instanceof ReorgOp
			|| hop instanceof DataOp || hop instanceof LiteralOp || hop instanceof NaryOp
			|| hop instanceof DataGenOp || hop instanceof FunctionOp )
			return OperatorType.NonAggregate;
		else if ( hop instanceof AggBinaryOp || hop instanceof AggUnaryOp  || hop instanceof UnaryOp )
			return OperatorType.Aggregate;
		else
			return null;
	}

	/**
	 * Propagate privacy constraints to output variables
	 * based on privacy constraint of CPOperand output in instruction
	 * which has been set during privacy propagation preprocessing.
	 * @param inst instruction for which privacy constraints are propagated
	 * @param ec execution context
	 */
	public static void postProcessInstruction(Instruction inst, ExecutionContext ec){
		// if inst has output
		List<CPOperand> instOutputs = getOutputOperands(inst);
		if (!instOutputs.isEmpty()){
			for ( CPOperand output : instOutputs ){
				PrivacyConstraint outputPrivacyConstraint = output.getPrivacyConstraint();
				if ( PrivacyUtils.someConstraintSetUnary(outputPrivacyConstraint) )
					setOutputPrivacyConstraint(ec, outputPrivacyConstraint, output.getName());
			}
		}
	}

	/**
	 * Propagate privacy constraints from input to output CPOperands
	 * in case the privacy constraints of the input are activated.
	 * @param inst instruction for which the privacy constraints are propagated
	 * @param ec execution context
	 * @return instruction with propagated privacy constraints (usually the same instance as the input inst)
	 */
	public static Instruction preprocessInstruction(Instruction inst, ExecutionContext ec){
		switch ( inst.getType() ){
			case CONTROL_PROGRAM:
				return preprocessCPInstruction( (CPInstruction) inst, ec );
			case BREAKPOINT:
			case SPARK:
			case GPU:
			case FEDERATED:
				return inst;
			default:
				return throwExceptionIfInputOrInstPrivacy(inst, ec);
		}
	}

	private static Instruction preprocessCPInstruction(CPInstruction inst, ExecutionContext ec){
		switch(inst.getCPInstructionType()){
			case Binary:
			case Builtin:
			case BuiltinNary:
			case FCall:
			case ParameterizedBuiltin:
			case Quaternary:
			case Reorg:
			case Ternary:
			case Unary:
			case MultiReturnBuiltin:
			case MultiReturnParameterizedBuiltin:
			case MatrixIndexing:
				return mergePrivacyConstraintsFromInput( inst, ec, OperatorType.NonAggregate );
			case AggregateTernary:
			case AggregateUnary:
				return mergePrivacyConstraintsFromInput(inst, ec, OperatorType.Aggregate);
			case Append:
				return preprocessAppendCPInstruction((AppendCPInstruction) inst, ec);
			case AggregateBinary:
				if ( inst instanceof AggregateBinaryCPInstruction )
					return preprocessAggregateBinaryCPInstruction((AggregateBinaryCPInstruction)inst, ec);
				else return throwExceptionIfInputOrInstPrivacy(inst, ec);
			case MMTSJ:
				OperatorType mmtsjOpType = OperatorType.getAggregationType((MMTSJCPInstruction) inst, ec);
				return mergePrivacyConstraintsFromInput(inst, ec, mmtsjOpType);
			case MMChain:
				OperatorType mmChainOpType = OperatorType.getAggregationType((MMChainCPInstruction) inst, ec);
				return mergePrivacyConstraintsFromInput(inst, ec, mmChainOpType);
			case Variable:
				return preprocessVariableCPInstruction((VariableCPInstruction) inst, ec);
			default:
				return throwExceptionIfInputOrInstPrivacy(inst, ec);
		}
	}

	private static Instruction preprocessVariableCPInstruction(VariableCPInstruction inst, ExecutionContext ec){
		switch ( inst.getVariableOpcode() ) {
			case CopyVariable:
			case MoveVariable:
			case RemoveVariableAndFile:
			case CastAsMatrixVariable:
			case CastAsFrameVariable:
			case Write:
			case SetFileName:
			case CastAsScalarVariable:
			case CastAsDoubleVariable:
			case CastAsIntegerVariable:
			case CastAsBooleanVariable:
				return propagateFirstInputPrivacy(inst, ec);
			case CreateVariable:
				return propagateSecondInputPrivacy(inst, ec);
			case AssignVariable:
			case RemoveVariable:
				return mergePrivacyConstraintsFromInput( inst, ec, OperatorType.NonAggregate );
			case Read:
				// Adds scalar object to variable map, hence input (data type and filename) privacy should not be propagated
				return inst;
			default:
				return throwExceptionIfInputOrInstPrivacy(inst, ec);
		}
	}

	/**
	 * Propagates fine-grained constraints if input has fine-grained constraints,
	 * otherwise it propagates general constraints.
	 * @param inst aggregate binary instruction for which constraints are propagated
	 * @param ec execution context
	 * @return instruction with merged privacy constraints propagated to it and output CPOperand
	 */
	private static Instruction preprocessAggregateBinaryCPInstruction(AggregateBinaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inst.getInputs());
		if ( PrivacyUtils.someConstraintSetBinary(privacyConstraints) ){
			PrivacyConstraint mergedPrivacyConstraint;
			if ( (privacyConstraints[0] != null && privacyConstraints[0].hasFineGrainedConstraints() ) ||
				(privacyConstraints[1] != null && privacyConstraints[1].hasFineGrainedConstraints() )){
				MatrixBlock input1 = ec.getMatrixInput(inst.input1.getName());
				MatrixBlock input2 = ec.getMatrixInput(inst.input2.getName());
				Propagator propagator = new MatrixMultiplicationPropagatorPrivateFirst(input1, privacyConstraints[0], input2, privacyConstraints[1]);
				mergedPrivacyConstraint = propagator.propagate();
				ec.releaseMatrixInput(inst.input1.getName(), inst.input2.getName());
			}
			else {
				mergedPrivacyConstraint = mergeNary(privacyConstraints, OperatorType.getAggregationType(inst, ec));
				inst.setPrivacyConstraint(mergedPrivacyConstraint);
			}
			inst.output.setPrivacyConstraint(mergedPrivacyConstraint);
		}
		return inst;
	}

	/**
	 * Propagates input privacy constraints using general and fine-grained constraints depending on the AppendType.
	 * @param inst append instruction for which constraints are propagated
	 * @param ec execution context
	 * @return instruction with merged privacy constraints propagated to it and output CPOperand
	 */
	private static Instruction preprocessAppendCPInstruction(AppendCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inst.getInputs());
		if ( PrivacyUtils.someConstraintSetBinary(privacyConstraints) ){
			if ( inst.getAppendType() == AppendCPInstruction.AppendType.STRING ){
				PrivacyLevel[] privacyLevels = new PrivacyLevel[2];
				privacyLevels[0] = PrivacyUtils.getGeneralPrivacyLevel(privacyConstraints[0]);
				privacyLevels[1] =  PrivacyUtils.getGeneralPrivacyLevel(privacyConstraints[1]);
				PrivacyConstraint outputConstraint = new PrivacyConstraint(corePropagation(privacyLevels, OperatorType.NonAggregate));
				inst.output.setPrivacyConstraint(outputConstraint);
			} else if ( inst.getAppendType() == AppendCPInstruction.AppendType.LIST ){
				ListObject input1 = (ListObject) ec.getVariable(inst.input1);
				if ( inst.getOpcode().equals("remove")){
					ScalarObject removePosition = ec.getScalarInput(inst.input2);
					PropagatorMultiReturn propagator = new ListRemovePropagator(input1, privacyConstraints[0], removePosition, removePosition.getPrivacyConstraint());
					PrivacyConstraint[] outputConstraints = propagator.propagate();
					inst.output.setPrivacyConstraint(outputConstraints[0]);
					((ListAppendRemoveCPInstruction) inst).getOutput2().setPrivacyConstraint(outputConstraints[1]);
				} else {
					ListObject input2 = (ListObject) ec.getVariable(inst.input2);
					Propagator propagator = new ListAppendPropagator(input1, privacyConstraints[0], input2, privacyConstraints[1]);
					inst.output.setPrivacyConstraint(propagator.propagate());
				}
			}
			else {
				MatrixBlock input1 = ec.getMatrixInput(inst.input1.getName());
				MatrixBlock input2 = ec.getMatrixInput(inst.input2.getName());
				Propagator propagator;
				if ( inst.getAppendType() == AppendCPInstruction.AppendType.RBIND )
					propagator = new RBindPropagator(input1, privacyConstraints[0], input2, privacyConstraints[1]);
				else if ( inst.getAppendType() == AppendCPInstruction.AppendType.CBIND )
					propagator = new CBindPropagator(input1, privacyConstraints[0], input2, privacyConstraints[1]);
				else throw new DMLPrivacyException("Instruction " + inst.getCPInstructionType() + " with append type " +
						inst.getAppendType() + " is not supported by the privacy propagator");
				inst.output.setPrivacyConstraint(propagator.propagate());
				ec.releaseMatrixInput(inst.input1.getName(), inst.input2.getName());
			}
		}
		return inst;
	}

	/**
	 * Propagates privacy constraints from input to instruction and output CPOperand based on given operator type.
	 * The propagation is done through the core propagation.
	 * @param inst instruction for which privacy is propagated
	 * @param ec execution context
	 * @param operatorType defining whether the instruction is aggregating the input
	 * @return instruction with the merged privacy constraint propagated to it and output CPOperand
	 */
	private static Instruction mergePrivacyConstraintsFromInput(Instruction inst, ExecutionContext ec,
		OperatorType operatorType){
		return mergePrivacyConstraintsFromInput(inst, ec, getInputOperands(inst), getOutputOperands(inst), operatorType);
	}

	/**
	 * Propagates privacy constraints from input to instruction and output CPOperand based on given operator type.
	 * The propagation is done through the core propagation.
	 * @param inst instruction for which privacy is propagated
	 * @param ec execution context
	 * @param inputs to instruction
	 * @param outputs of instruction
	 * @param operatorType defining whether the instruction is aggregating the input
	 * @return instruction with the merged privacy constraint propagated to it and output CPOperand
	 */
	private static Instruction mergePrivacyConstraintsFromInput(Instruction inst, ExecutionContext ec,
		CPOperand[] inputs, List<CPOperand> outputs, OperatorType operatorType){
		if ( inputs != null && inputs.length > 0 ){
			PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inputs);
			if ( privacyConstraints != null ){
				PrivacyConstraint mergedPrivacyConstraint = mergeNary(privacyConstraints, operatorType);
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

	/**
	 * Throw exception if privacy constraint activated for instruction or for input to instruction.
	 * @param inst covariance instruction
	 * @param ec execution context
	 * @return input instruction if privacy constraints are not activated
	 */
	private static Instruction throwExceptionIfInputOrInstPrivacy(Instruction inst, ExecutionContext ec){
		throwExceptionIfPrivacyActivated(inst);
		CPOperand[] inputOperands = getInputOperands(inst);
		if (inputOperands != null){
			for ( CPOperand input : inputOperands ){
				PrivacyConstraint privacyConstraint = getInputPrivacyConstraint(ec, input);
				if ( privacyConstraint != null && privacyConstraint.hasConstraints()){
					throw new DMLPrivacyException("Input of instruction " + inst + " has privacy constraints activated, but the constraints are not propagated during preprocessing of instruction.");
				}
			}
		}
		return inst;
	}

	private static void throwExceptionIfPrivacyActivated(Instruction inst){
		if ( inst.getPrivacyConstraint() != null && inst.getPrivacyConstraint().hasConstraints() ) {
			throw new DMLPrivacyException("Instruction " + inst + " has privacy constraints activated, but the constraints are not propagated during preprocessing of instruction.");
		}
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

	/**
	 * Returns input privacy constraints as array or returns null if no privacy constraints are found in the inputs.
	 * @param ec execution context
	 * @param inputs from which privacy constraints are retrieved
	 * @return array of privacy constraints from inputs
	 */
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

	/**
	 * Returns input CPOperands of instruction or returns null if instruction type is not supported by this method.
	 * @param inst instruction from which the inputs are retrieved
	 * @return array of input CPOperands or null
	 */
	private static CPOperand[] getInputOperands(Instruction inst){
		if ( inst instanceof ComputationCPInstruction )
			return ((ComputationCPInstruction)inst).getInputs();
		if ( inst instanceof BuiltinNaryCPInstruction )
			return ((BuiltinNaryCPInstruction)inst).getInputs();
		if ( inst instanceof FunctionCallCPInstruction )
			return ((FunctionCallCPInstruction)inst).getInputs();
		if ( inst instanceof SqlCPInstruction )
			return ((SqlCPInstruction)inst).getInputs();
		else return null;
	}

	/**
	 * Returns a list of output CPOperands of instruction or an empty list if the instruction has no outputs.
	 * Note that this method needs to be extended as new instruction types are added, otherwise it will
	 * return an empty list for instructions that may have outputs.
	 * @param inst instruction from which the outputs are retrieved
	 * @return list of outputs
	 */
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
		else if ( inst instanceof BuiltinNaryCPInstruction )
			return getSingletonList(((BuiltinNaryCPInstruction)inst).getOutput());
		return new ArrayList<>();
	}

	private static List<CPOperand> getSingletonList(CPOperand operand){
		if ( operand != null)
			return new ArrayList<>(Collections.singletonList(operand));
		return new ArrayList<>();
	}
}
