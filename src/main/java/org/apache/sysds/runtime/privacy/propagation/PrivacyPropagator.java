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

import java.util.*;

import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.Hop;
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
	public static Data parseAndSetPrivacyConstraint(Data cd, JSONObject mtd)
		throws JSONException
	{
		PrivacyConstraint mtdPrivConstraint = parseAndReturnPrivacyConstraint(mtd);
		if ( mtdPrivConstraint != null )
			cd.setPrivacyConstraints(mtdPrivConstraint);
		return cd;
	}

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
		PrivacyConstraint[] inputConstraints = hop.getInput().stream()
			.map(Hop::getPrivacy).toArray(PrivacyConstraint[]::new);
		if ( hop instanceof TernaryOp || hop instanceof BinaryOp || hop instanceof ReorgOp )
			hop.setPrivacy(mergeNary(inputConstraints, OperatorType.NonAggregate));
		else if ( hop instanceof AggBinaryOp || hop instanceof AggUnaryOp  || hop instanceof UnaryOp )
			hop.setPrivacy(mergeNary(inputConstraints, OperatorType.Aggregate));
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
				throwExceptionIfPrivacyActivated(inst);
				return inst;
		}
	}

	private static Instruction preprocessCPInstruction(CPInstruction inst, ExecutionContext ec){
		CPOperand[] inputOperands = getInputOperands(inst);
		List<CPOperand> outputOperands = getOutputOperands(inst);

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
				return mergePrivacyConstraintsFromInput( inst, ec, inputOperands, outputOperands, OperatorType.NonAggregate );
			case AggregateTernary:
			case AggregateUnary:
				return mergePrivacyConstraintsFromInput(inst, ec, inputOperands, outputOperands, OperatorType.Aggregate);
			case Append:
				return preprocessAppendCPInstruction((AppendCPInstruction) inst, ec);
			case AggregateBinary:
				if ( inst instanceof AggregateBinaryCPInstruction ){
					// This can only be a matrix multiplication and it does not count as an aggregation in terms of privacy.
					return preprocessAggregateBinaryCPInstruction((AggregateBinaryCPInstruction)inst, ec);
				} else if ( inst instanceof CovarianceCPInstruction ){
					return throwExceptionIfInputOrInstPrivacy((CovarianceCPInstruction)inst, ec, inputOperands);
				} else return preprocessInstructionSimple(inst, ec);
			case MultiReturnBuiltin:
			case MultiReturnParameterizedBuiltin:
				return mergePrivacyConstraintsFromInput(inst, ec, inputOperands, getOutputOperands(inst), OperatorType.NonAggregate);
			case Variable:
				return preprocessVariableCPInstruction((VariableCPInstruction) inst, ec, inputOperands, outputOperands);
			default:
				return preprocessInstructionSimple(inst, ec);
		}
	}

	private static Instruction preprocessVariableCPInstruction(VariableCPInstruction inst, ExecutionContext ec,
		CPOperand[] inputOperands, List<CPOperand> outputOperands){
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
				return mergePrivacyConstraintsFromInput( inst, ec, inputOperands, outputOperands, OperatorType.NonAggregate );
			case Read:
				return inst;
			default:
				throwExceptionIfPrivacyActivated(inst);
				return inst;
		}
	}

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
				mergedPrivacyConstraint = mergeNary(privacyConstraints, OperatorType.Aggregate);
				inst.setPrivacyConstraint(mergedPrivacyConstraint);
			}
			inst.output.setPrivacyConstraint(mergedPrivacyConstraint);
		}
		return inst;
	}

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
				Propagator propagator = null;
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
	 * Throw exception if privacy constraints are activated or return instruction if privacy is not activated
	 * @param inst instruction
	 * @param ec execution context
	 * @return instruction
	 */
	private static Instruction preprocessInstructionSimple(Instruction inst, ExecutionContext ec){
		throwExceptionIfPrivacyActivated(inst);
		return inst;
	}

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
	private static Instruction throwExceptionIfInputOrInstPrivacy(Instruction inst, ExecutionContext ec, CPOperand[] inputOperands){
		throwExceptionIfPrivacyActivated(inst);
		for ( CPOperand input : inputOperands ){
			PrivacyConstraint privacyConstraint = getInputPrivacyConstraint(ec, input);
			if ( privacyConstraint != null){
				throw new DMLPrivacyException("Input of instruction " + inst + " has privacy constraints activated, but the constraints are not propagated during preprocessing of instruction.");
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

	private static CPOperand[] getInputOperands(Instruction inst){
		if ( inst instanceof ComputationCPInstruction )
			return ((ComputationCPInstruction)inst).getInputs();
		else return null;
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
