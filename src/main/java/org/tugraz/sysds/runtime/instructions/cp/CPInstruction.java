/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.instructions.cp;

import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.lops.Lop;
import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.data.TensorBlock;
import org.tugraz.sysds.runtime.instructions.CPInstructionParser;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.Operator;
import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;

public abstract class CPInstruction extends Instruction 
{
	private static final String DELIM = " ";

	public enum CPType {
		AggregateUnary, AggregateBinary, AggregateTernary,
		Unary, Binary, Ternary, Quaternary, BuiltinNary, Ctable, 
		MultiReturnParameterizedBuiltin, ParameterizedBuiltin, MultiReturnBuiltin,
		Builtin, Reorg, Variable, External, Append, Rand, QSort, QPick,
		MatrixIndexing, MMTSJ, PMMJ, MMChain, Reshape, Partition, Compression, SpoofFused,
		StringInit, CentralMoment, Covariance, UaggOuterChain, Dnn }
	
	protected final CPType _cptype;
	protected final Operator _optr;
	protected final boolean _requiresLabelUpdate;
	
	// Generic miscellaneous timers that are applicable to all CP (and few SP) instructions 
	public final static String MISC_TIMER_GET_SPARSE_MB =          		"aqrs";	// time spent in bringing input sparse matrix block
	public final static String MISC_TIMER_GET_DENSE_MB =          		"aqrd";	// time spent in bringing input dense matrix block
	public final static String MISC_TIMER_ACQ_MODIFY_SPARSE_MB =        "aqms";	// time spent in bringing output sparse matrix block
	public final static String MISC_TIMER_ACQ_MODIFY_DENSE_MB =         "aqmd";	// time spent in bringing output dense matrix block
	public final static String MISC_TIMER_RELEASE_INPUT_MB =      		"rlsi";	// time spent in release input matrix block
	public final static String MISC_TIMER_RELEASE_EVICTION =			"rlsev";// time spent in buffer eviction of release operation
	public final static String MISC_TIMER_RELEASE_BUFF_WRITE =			"rlswr";// time spent in buffer write in release operation
	public final static String MISC_TIMER_SPARSE_TO_DENSE =				"s2d";  // time spent in sparse to dense conversion
	public final static String MISC_TIMER_DENSE_TO_SPARSE =				"d2s";  // time spent in sparse to dense conversion
	public final static String MISC_TIMER_RECOMPUTE_NNZ =				"rnnz"; // time spent in recompute non-zeroes
	
	// Instruction specific miscellaneous timers that were found as potential bottlenecks in one of performance analysis.
	// SystemML committers have to be judicious about adding them by weighing the tradeoffs between reuse in future analysis and unnecessary overheads.
	public final static String MISC_TIMER_CSR_LIX_COPY =				"csrlix";// time spent in CSR-specific method to address performance issues due to repeated re-shifting on update-in-place.
	public final static String MISC_TIMER_LIX_COPY =					"lixcp";// time spent in range copy

	protected CPInstruction(CPType type, String opcode, String istr) {
		this(type, null, opcode, istr);
	}

	protected CPInstruction(CPType type, Operator op, String opcode, String istr) {
		_cptype = type;
		_optr = op;
		instString = istr;

		// prepare opcode and update requirement for repeated usage
		instOpcode = opcode;
		_requiresLabelUpdate = super.requiresLabelUpdate();
	}
	
	@Override
	public IType getType() {
		return IType.CONTROL_PROGRAM;
	}

	public CPType getCPInstructionType() {
		return _cptype;
	}
	
	@Override
	public boolean requiresLabelUpdate() {
		return _requiresLabelUpdate;
	}

	@Override
	public String getGraphString() {
		return getOpcode();
	}

	@Override
	public Instruction preprocessInstruction(ExecutionContext ec) {
		//default preprocess behavior (e.g., debug state)
		Instruction tmp = super.preprocessInstruction(ec);
		
		//instruction patching
		if( tmp.requiresLabelUpdate() ) { //update labels only if required
			//note: no exchange of updated instruction as labels might change in the general case
			String updInst = updateLabels(tmp.toString(), ec.getVariables());
			tmp = CPInstructionParser.parseSingleInstruction(updInst);
			// Corrected lineage trace for patched instructions
			if (DMLScript.LINEAGE)
				ec.traceLineage(tmp);
		}
		return tmp;
	}

	@Override 
	public abstract void processInstruction(ExecutionContext ec);
	
	/**
	 * Takes a delimited string of instructions, and replaces ALL placeholder labels 
	 * (such as ##mVar2## and ##Var5##) in ALL instructions.
	 *  
	 * @param instList instruction list as string
	 * @param labelValueMapping local variable map
	 * @return instruction list after replacement
	 */
	public static String updateLabels (String instList, LocalVariableMap labelValueMapping) {

		if ( !instList.contains(Lop.VARIABLE_NAME_PLACEHOLDER) )
			return instList;
		
		StringBuilder updateInstList = new StringBuilder();
		String[] ilist = instList.split(Lop.INSTRUCTION_DELIMITOR); 
		
		for ( int i=0; i < ilist.length; i++ ) {
			if ( i > 0 )
				updateInstList.append(Lop.INSTRUCTION_DELIMITOR);
			
			updateInstList.append( updateInstLabels(ilist[i], labelValueMapping));
		}
		return updateInstList.toString();
	}
	
	/** 
	 * Replaces ALL placeholder strings (such as ##mVar2## and ##Var5##) in a single instruction.
	 *  
	 * @param inst string instruction
	 * @param map local variable map
	 * @return string instruction after replacement
	 */
	private static String updateInstLabels(String inst, LocalVariableMap map) {
		if ( inst.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ) {
			int skip = Lop.VARIABLE_NAME_PLACEHOLDER.length();
			while ( inst.contains(Lop.VARIABLE_NAME_PLACEHOLDER) ) {
				int startLoc = inst.indexOf(Lop.VARIABLE_NAME_PLACEHOLDER)+skip;
				String varName = inst.substring(startLoc, inst.indexOf(Lop.VARIABLE_NAME_PLACEHOLDER, startLoc));
				String replacement = getVarNameReplacement(inst, varName, map);
				inst = inst.replaceAll(Lop.VARIABLE_NAME_PLACEHOLDER + varName + Lop.VARIABLE_NAME_PLACEHOLDER, replacement);
			}
		}
		return inst;
	}
	
	/**
	 * Computes the replacement string for a given variable name placeholder string 
	 * (e.g., ##mVar2## or ##Var5##). The replacement is a HDFS filename for matrix 
	 * variables, and is the actual value (stored in symbol table) for scalar variables.
	 * 
	 * @param inst instruction
	 * @param varName variable name
	 * @param map local variable map
	 * @return string variable name
	 */
	private static String getVarNameReplacement(String inst, String varName, LocalVariableMap map) {
		Data val = map.get(varName);
		if (val != null) {
			String replacement = null;
			if (val.getDataType() == DataType.MATRIX) {
				replacement = ((MatrixObject) val).getFileName();
			}

			if (val.getDataType() == DataType.SCALAR)
				replacement = "" + ((ScalarObject) val).getStringValue();
			return replacement;
		} else {
			throw new DMLRuntimeException("Variable (" + varName + ") in Instruction (" + inst + ") is not found in the variablemap.");
		}
	}

	public  int[] getTensorDimensions(ExecutionContext ec, CPOperand dims) {
		int[] tDims;
		switch (dims.getDataType()) {
			case SCALAR: {
				// Dimensions given as string
				if (dims.getValueType() != Types.ValueType.STRING) {
					throw new DMLRuntimeException("Dimensions have to be passed as list, string, matrix or tensor.");
				}
				String dimensionString = ec.getScalarInput(dims.getName(), Types.ValueType.STRING, dims.isLiteral())
						.getStringValue();
				StringTokenizer dimensions = new StringTokenizer(dimensionString, DELIM);
				tDims = new int[dimensions.countTokens()];
				Arrays.setAll(tDims, (i) -> Integer.parseInt(dimensions.nextToken()));
			}
			break;
			case MATRIX: {
				// Dimensions given as vector
				MatrixBlock in = ec.getMatrixInput(dims.getName());
				boolean colVec = false;
				if (in.getNumRows() == 1) {
					colVec = true;
				} else if (!(in.getNumColumns() == 1)) {
					throw new DMLRuntimeException("Dimensions matrix has to be a vector.");
				}
				tDims = new int[(int) in.getLength()];
				for (int i = 0; i < in.getLength(); i++) {
					tDims[i] = UtilFunctions.toInt(in.getValue(colVec ? 0 : i, colVec ? i : 0));
				}
				ec.releaseMatrixInput(dims.getName());
			}
			break;
			case TENSOR: {
				// Dimensions given as vector
				TensorBlock in = ec.getTensorInput(dims.getName());
				boolean colVec = false;
				if (!in.isVector()) {
					throw new DMLRuntimeException("Dimensions tensor has to be a vector.");
				} else if (in.getNumRows() == 1) {
					colVec = true;
				}
				tDims = new int[(int) in.getLength()];
				for (int i = 0; i < in.getLength(); i++) {
					tDims[i] = UtilFunctions.toInt(in.get(new int[]{colVec ? 0 : i, colVec ? i : 0}));
				}
				ec.releaseTensorInput(dims.getName());
			}
			break;
			case LIST: {
				// Dimensions given as List
				ListObject list = ec.getListObject(dims.getName());
				tDims = new int[list.getLength()];
				List<Data> dimsData = list.getData();
				for (int i = 0; i < tDims.length; i++) {
					if (dimsData.get(i) instanceof ScalarObject) {
						// TODO warning if double value is cast to long?
						tDims[i] = (int) ((ScalarObject) dimsData.get(i)).getLongValue();
					} else {
						throw new DMLRuntimeException("Dims parameter for does not support lists with non scalar values.");
					}
				}
			}
			break;
			default:
				throw new DMLRuntimeException("Dimensions have to be passed as list, string, matrix or tensor.");
		}
		return tDims;
	}
}
