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

package org.tugraz.sysds.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.tugraz.sysds.parser.DataIdentifier;
import org.tugraz.sysds.parser.ExternalFunctionStatement;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.udf.ExternalFunctionInvocationInstruction;
import org.tugraz.sysds.udf.PackageFunction;
import org.tugraz.sysds.udf.Scalar.ScalarValueType;

public class ExternalFunctionProgramBlock extends FunctionProgramBlock 
{
	protected static final IDSequence _idSeq = new IDSequence();

	protected long _runID = -1; //ID for block of statements
	protected String _baseDir = null;
	protected HashMap<String, String> _otherParams; // holds other key value parameters 

	private ArrayList<Instruction> block2CellInst; 
	private ArrayList<Instruction> cell2BlockInst; 
	
	/**
	 * Constructor that also provides otherParams that are needed for external
	 * functions. Remaining parameters will just be passed to constructor for
	 * function program block.
	 * 
	 * @param prog runtime program
	 * @param inputParams list of input data identifiers
	 * @param outputParams list of output data indentifiers
	 * @param baseDir base directory
	 */
	protected ExternalFunctionProgramBlock(Program prog,
			ArrayList<DataIdentifier> inputParams,
			ArrayList<DataIdentifier> outputParams,
			String baseDir)
	{
		super(prog, inputParams, outputParams);
		_baseDir = baseDir;
	}
	
	public ExternalFunctionProgramBlock(Program prog,
			ArrayList<DataIdentifier> inputParams,
			ArrayList<DataIdentifier> outputParams,
			HashMap<String, String> otherParams,
			String baseDir) {

		super(prog, inputParams, outputParams);
		_baseDir = baseDir;
		
		// copy other params
		_otherParams = new HashMap<>();
		_otherParams.putAll(otherParams);

		// generate instructions
		createInstructions();
	}
	
	public String getBaseDir() {
		return _baseDir;
	}
	
	public HashMap<String,String> getOtherParams() {
		return _otherParams;
	}
	
	/**
	 * Method to be invoked to execute instructions for the external function
	 * invocation
	 */
	@Override
	public void execute(ExecutionContext ec) 
	{
		_runID = _idSeq.getNextID();
		
		// export input variables to HDFS (see RunMRJobs)
		ArrayList<DataIdentifier> inputParams = null;
		
		try {
			inputParams = getInputParams();
			for(DataIdentifier di : inputParams ) {
				Data d = ec.getVariable(di.getName());
				if( d.getDataType().isMatrix() )
					((MatrixObject) d).exportData();
			}
		}
		catch (Exception e){
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error exporting input variables to HDFS", e);
		}
		
		// convert block to cell
		if( block2CellInst != null )
		{
			ArrayList<Instruction> tempInst = new ArrayList<>();
			tempInst.addAll(block2CellInst);
			try {
				this.executeInstructions(tempInst,ec);
			} catch (Exception e) {
				
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error executing "
						+ tempInst.toString(), e);
			}
		}
		
		// now execute package function
		for (int i = 0; i < _inst.size(); i++) 
		{
			try {
				if (_inst.get(i) instanceof ExternalFunctionInvocationInstruction)
					((ExternalFunctionInvocationInstruction) _inst.get(i)).processInstruction(ec);
			} 
			catch(Exception e) {
				throw new DMLRuntimeException(this.printBlockErrorLocation() + 
						"Failed to execute instruction " + _inst.get(i).toString(), e);
			}
		}

		// convert cell to block
		if( cell2BlockInst != null )
		{
			ArrayList<Instruction> tempInst = new ArrayList<>();
			try {
				tempInst.clear();
				tempInst.addAll(cell2BlockInst);
				this.executeInstructions(tempInst, ec);
			} catch (Exception e) {
				
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Failed to execute instruction "
						+ cell2BlockInst.toString(), e);
			}
		}
		
		// check return values
		checkOutputParameters(ec.getVariables());
	}

	/**
	 * Given a list of parameters as data identifiers, returns an array
	 * of instruction operands.
	 * 
	 * @param params list of data identifiers
	 * @return operands
	 */
	protected CPOperand[] getOperands(ArrayList<DataIdentifier> params) {
		CPOperand[] ret = new CPOperand[params.size()];
		for (int i = 0; i < params.size(); i++) {
			DataIdentifier param = params.get(i);
			ret[i] = new CPOperand(param.getName(),
				param.getValueType(), param.getDataType());
		}
		return ret;
	}

	/**
	 * method to create instructions
	 * 
	 */
	protected void createInstructions() {

		_inst = new ArrayList<>();

		// unblock all input matrices
		block2CellInst = null;

		// assemble information provided through keyvalue pairs
		String className = _otherParams.get(ExternalFunctionStatement.CLASS_NAME);
		String configFile = _otherParams.get(ExternalFunctionStatement.CONFIG_FILE);
		
		// class name cannot be null, however, configFile and execLocation can be null
		if (className == null)
			throw new RuntimeException(this.printBlockErrorLocation() + ExternalFunctionStatement.CLASS_NAME + " not provided!");

		// assemble input and output operands
		CPOperand[] inputs = getOperands(getInputParams());
		CPOperand[] outputs = getOperands(getOutputParams());
		
		// generate instruction
		PackageFunction fun = createFunctionObject(className, configFile);
		ExternalFunctionInvocationInstruction einst = 
			new ExternalFunctionInvocationInstruction(inputs, outputs, fun, _baseDir, InputInfo.TextCellInputInfo);
		verifyFunctionInputsOutputs(fun, inputs, outputs);
		if (getInputParams().size() > 0)
			einst.setLocation(getInputParams().get(0));
		else if (getOutputParams().size() > 0)
			einst.setLocation(getOutputParams().get(0));
		else
			einst.setLocation(getFilename(), _beginLine, _endLine, _beginColumn, _endColumn);
		_inst.add(einst);

		// block output matrices
		cell2BlockInst = null;
	}
	
	@SuppressWarnings("unchecked")
	protected PackageFunction createFunctionObject(String className, String configFile) {
		try {
			//create instance of package function
			Class<Instruction> cla = (Class<Instruction>) Class.forName(className);
			Object o = cla.newInstance();
			if (!(o instanceof PackageFunction))
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Class is not of type PackageFunction");
			PackageFunction fun = (PackageFunction) o;
			
			//configure package function
			fun.setConfiguration(configFile);
			fun.setBaseDir(_baseDir);
			
			return fun;
		} 
		catch (Exception e) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error instantiating package function ", e );
		}
	}
	
	protected void verifyFunctionInputsOutputs(PackageFunction fun, CPOperand[] inputs, CPOperand[] outputs) {
		// verify number of outputs if fixed, otherwise best effort handle of outputs
		if( !fun.hasVarNumFunctionOutputs()
			&& outputs.length != fun.getNumFunctionOutputs() ) {
			throw new DMLRuntimeException(
					"Number of function outputs ("+fun.getNumFunctionOutputs()+") " +
					"does not match with declaration ("+outputs.length+").");
		}
	}

	
	
	/**
	 * Method to get string representation of scalar value type
	 * 
	 * @param scalarType scalar value type
	 * @return scalar value type string
	 */
	protected String getScalarValueTypeString(ScalarValueType scalarType) {
		if (scalarType.equals(ScalarValueType.Text))
			return "String";
		else
			return scalarType.toString();
	}
	
	@Override
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in external function program block generated from external function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}