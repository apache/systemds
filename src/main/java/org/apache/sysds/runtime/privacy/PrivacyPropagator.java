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

import java.util.function.Function;

import org.apache.directory.api.util.exception.NotImplementedException;
import org.apache.sysds.runtime.DMLPrivacyException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.BinaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.BuiltinNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPInstruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.MultiReturnParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ParameterizedBuiltinCPInstruction;
import org.apache.sysds.runtime.instructions.cp.QuaternaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.TernaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.UnaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;

/**
 * Class with static methods merging privacy constraints of operands 
 * in expressions to generate the privacy constraints of the output. 
 */
public class PrivacyPropagator {

	public static PrivacyConstraint mergeBinary(PrivacyConstraint privacyConstraint1, PrivacyConstraint privacyConstraint2) {
		if (privacyConstraint1 != null && privacyConstraint2 != null)
			return new PrivacyConstraint(
				privacyConstraint1.getPrivacy() || privacyConstraint2.getPrivacy());
		else if (privacyConstraint1 != null)
			return privacyConstraint1;
		else if (privacyConstraint2 != null)
			return privacyConstraint2; 
		return null;
	}

	public static PrivacyConstraint mergeTernary(PrivacyConstraint[] privacyConstraints){
		return mergeBinary(mergeBinary(privacyConstraints[0], privacyConstraints[1]), privacyConstraints[2]);
	}

	public static PrivacyConstraint mergeQuaternary(PrivacyConstraint[] privacyConstraints){
		return mergeBinary(mergeTernary(privacyConstraints), privacyConstraints[3]);
	}

	public static PrivacyConstraint mergeNary(PrivacyConstraint[] privacyConstraints){
		PrivacyConstraint mergedPrivacyConstraint = privacyConstraints[0];
		for ( int i = 1; i < privacyConstraints.length; i++ ){
			mergedPrivacyConstraint = mergeBinary(mergedPrivacyConstraint, privacyConstraints[i]);
		}
		return mergedPrivacyConstraint;
	}

	public static Instruction preprocessCPInstruction(CPInstruction inst, ExecutionContext ec){
		switch ( inst.getCPInstructionType() )
		{
			case Variable:
				return preprocessVariableCPInstruction((VariableCPInstruction) inst, ec);
			case AggregateUnary:
			case Reorg:
			case Unary:
				return preprocessUnaryCPInstruction((UnaryCPInstruction) inst, ec);
			case AggregateBinary:
			case Append: 
			case Binary:
				return preprocessBinaryCPInstruction((BinaryCPInstruction) inst, ec);
			case AggregateTernary: 
			case Ternary:
				return preprocessTernaryCPInstruction((TernaryCPInstruction) inst, ec);
			case Quaternary: 
				return preprocessQuaternary((QuaternaryCPInstruction) inst, ec);
			case BuiltinNary:
			case Builtin:
				return preprocessBuiltinNary((BuiltinNaryCPInstruction) inst, ec);
			case Ctable: 
				return preprocessCtableInstruction((CtableCPInstruction) inst, ec);
			case MultiReturnParameterizedBuiltin:
			case MultiReturnBuiltin:  
				return preprocessMultiReturnParameterizedBuiltinInstruction((MultiReturnParameterizedBuiltinCPInstruction) inst, ec);
			case ParameterizedBuiltin:
				return preprocessParameterizedBuiltinInstruction((ParameterizedBuiltinCPInstruction) inst, ec);
			default:
				throwExceptionIfPrivacyActivated(inst, ec);
				return inst;
		}
	}

	public static Instruction preprocessParameterizedBuiltinInstruction(ParameterizedBuiltinCPInstruction inst, ExecutionContext ec){
		throw new NotImplementedException();
	}

	public static Instruction preprocessMultiReturnParameterizedBuiltinInstruction(MultiReturnParameterizedBuiltinCPInstruction inst, ExecutionContext ec){
		throw new NotImplementedException();
	}

	public static Instruction preprocessCtableInstruction(CtableCPInstruction inst, ExecutionContext ec){
		throw new NotImplementedException();
	}

	public static Instruction preprocessBuiltinNary(BuiltinNaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inst.getInputs());
		PrivacyConstraint mergedPrivacyConstraint = mergeNary(privacyConstraints);
		inst.setPrivacyConstraint(mergedPrivacyConstraint);
		setOutputPrivacyConstraint(ec, mergedPrivacyConstraint, inst.getOutput());
		return inst;
	}

	public static Instruction preprocessQuaternary(QuaternaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, new CPOperand[] {inst.input1,inst.input2,inst.input3,inst.input4});
		PrivacyConstraint mergedPrivacyConstraint = mergeQuaternary(privacyConstraints);
		inst.setPrivacyConstraint(mergedPrivacyConstraint);
		setOutputPrivacyConstraint(ec, mergedPrivacyConstraint, inst.output);
		return inst;
	}

	public static Instruction preprocessTernaryCPInstruction(TernaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, new CPOperand[]{inst.input1, inst.input2, inst.input3});
		PrivacyConstraint mergedPrivacyConstraint = mergeTernary(privacyConstraints);
		inst.setPrivacyConstraint(mergedPrivacyConstraint);
		setOutputPrivacyConstraint(ec, mergedPrivacyConstraint, inst.output);
		return inst;
	}

	public static Instruction preprocessNaryInstruction(CPInstruction inst, ExecutionContext ec, CPOperand[] inputs, CPOperand output, Function<PrivacyConstraint[], PrivacyConstraint> mergeFunction){
		PrivacyConstraint[] privacyConstraints = getInputPrivacyConstraints(ec, inputs);
		PrivacyConstraint mergedPrivacyConstraint = mergeFunction.apply(privacyConstraints);
		inst.setPrivacyConstraint(mergedPrivacyConstraint);
		setOutputPrivacyConstraint(ec, mergedPrivacyConstraint, output);
		return inst;

	}

	public static Instruction preprocessBinaryCPInstruction(BinaryCPInstruction inst, ExecutionContext ec){
		PrivacyConstraint privacyConstraint1 = getInputPrivacyConstraint(ec, inst.input1);
		PrivacyConstraint privacyConstraint2 = getInputPrivacyConstraint(ec, inst.input2);
		PrivacyConstraint mergedPrivacyConstraint = mergeBinary(privacyConstraint1, privacyConstraint2);
		inst.setPrivacyConstraint(mergedPrivacyConstraint);
		setOutputPrivacyConstraint(ec, mergedPrivacyConstraint, inst.output);
		return inst;
	}

	public static Instruction preprocessUnaryCPInstruction(UnaryCPInstruction inst, ExecutionContext ec){
		return propagateInputPrivacy(inst, ec, inst.input1, inst.output);
	}

	public static Instruction preprocessVariableCPInstruction(VariableCPInstruction inst, ExecutionContext ec){
		switch ( inst.getOpcodeVariable() )
		{
			case CreateVariable:
				return propagateSecondInputPrivacy(inst, ec);
			case AssignVariable:
				//Assigns scalar, hence it does not have privacy activated
				return inst;
			case CopyVariable:
			case MoveVariable:
				return propagateFirstInputPrivacy(inst, ec);
			case RemoveVariable:
				return propagateAllInputPrivacy(inst, ec);
			case RemoveVariableAndFile:
				return propagateFirstInputPrivacy(inst, ec);
			case CastAsScalarVariable: 
				return propagateCastAsScalarVariablePrivacy(inst, ec);
			case CastAsMatrixVariable:
			case CastAsFrameVariable:
				return propagateFirstInputPrivacy(inst, ec);
			case CastAsDoubleVariable:
			case CastAsIntegerVariable:
			case CastAsBooleanVariable:
				return propagateCastAsScalarVariablePrivacy(inst, ec);
			case Read:
				return inst;
			case Write:
				inst.setPrivacyConstraint(ec.getMatrixObject(inst.getInput1().getName()).getPrivacyConstraint() );
				return inst;
			case SetFileName:
				return propagateFirstInputPrivacy(inst, ec);
			default:
				throwExceptionIfPrivacyActivated(inst, ec);
				return inst;
		}
	}

	private static void throwExceptionIfPrivacyActivated(Instruction inst, ExecutionContext ec){
		if ( inst.getPrivacyConstraint() != null ){
			throw new DMLPrivacyException("Instruction " + inst + " has privacy constraints activated, but the constraints are not propagated during preprocessing of instruction.");
		}
	}

	/**
	 * Propagate privacy from first input and throw exception if privacy is activated.
	 * @param inst Instruction
	 * @param ec execution context
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateCastAsScalarVariablePrivacy(VariableCPInstruction inst, ExecutionContext ec){
		inst = (VariableCPInstruction) propagateFirstInputPrivacy(inst, ec);
		throwExceptionIfPrivacyActivated(inst, ec);
		return inst;
	}

	/**
	 * Propagate privacy constraints from all inputs if privacy constraints are set.
	 * @param inst instruction
	 * @param ec execution context
	 * @return instruction with or without privacy constraints
	 */
	private static Instruction propagateAllInputPrivacy(VariableCPInstruction inst, ExecutionContext ec){
		//TODO: Propagate the most restricting constraints instead of just the latest activated constraint
		for ( CPOperand input : inst.getInputs() )
			inst = (VariableCPInstruction) propagateInputPrivacy(inst, ec, input, inst.getOutput());
		return inst;
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
			setOutputPrivacyConstraint(ec, privacyConstraint, outputOperand);
		}
		return inst;
	}

	private static PrivacyConstraint getInputPrivacyConstraint(ExecutionContext ec, CPOperand input){
		Data dd = ec.getVariable(input);
		if ( dd != null && dd instanceof CacheableData)
			return ((CacheableData<?>) dd).getPrivacyConstraint();
		return null;
	}


	private static PrivacyConstraint[] getInputPrivacyConstraints(ExecutionContext ec, CPOperand[] inputs){
		PrivacyConstraint[] privacyConstraints = new PrivacyConstraint[inputs.length];
		for ( int i = 0; i < inputs.length; i++ ){
			privacyConstraints[i] = getInputPrivacyConstraint(ec, inputs[i]);
		}
		return privacyConstraints;
	}

	private static void setOutputPrivacyConstraint(ExecutionContext ec, PrivacyConstraint privacyConstraint, CPOperand output){
		Data dd = ec.getVariable(output);
		if ( dd != null && dd instanceof CacheableData ){
			((CacheableData<?>) dd).setPrivacyConstraints(privacyConstraint);
			ec.setVariable(output.getName(), dd);
		}
		else throw new DMLPrivacyException("Privacy constraint of " + output + " cannot be set since it is not an instance of CacheableData");
	}
}