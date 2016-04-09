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

package org.apache.sysml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.ExternalFunctionStatement;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.ExternalFunctionInvocationInstruction;
import org.apache.sysml.udf.Matrix;

/**
 * CP external function program block, that overcomes the need for 
 * BlockToCell and CellToBlock MR jobs by changing the contract for an external function.
 * If execlocation="CP", the implementation of an external function must read and write
 * matrices as InputInfo.BinaryBlockInputInfo and OutputInfo.BinaryBlockOutputInfo.
 * 
 * Furthermore, it extends ExternalFunctionProgramBlock with a base directory in order
 * to make it parallelizable, even in case of different JVMs. For this purpose every
 * external function must implement a <SET_BASE_DIR> method. 
 * 
 *
 */
public class ExternalFunctionProgramBlockCP extends ExternalFunctionProgramBlock 
{
	
	public static String DEFAULT_FILENAME = "ext_funct";
	private static IDSequence _defaultSeq = new IDSequence();
	
	/**
	 * Constructor that also provides otherParams that are needed for external
	 * functions. Remaining parameters will just be passed to constructor for
	 * function program block.
	 * 
	 * @param eFuncStat
	 * @throws DMLRuntimeException 
	 */
	public ExternalFunctionProgramBlockCP(Program prog,
			ArrayList<DataIdentifier> inputParams,
			ArrayList<DataIdentifier> outputParams,
			HashMap<String, String> otherParams,
			String baseDir) throws DMLRuntimeException {

		super(prog, inputParams, outputParams, baseDir); //w/o instruction generation
		
		// copy other params 
		_otherParams = new HashMap<String, String>();
		_otherParams.putAll(otherParams);

		// generate instructions (overwritten)
		createInstructions();
	}

	/**
	 * Method to be invoked to execute instructions for the external function
	 * invocation
	 * @throws DMLRuntimeException 
	 */
	@Override
	public void execute(ExecutionContext ec) throws DMLRuntimeException 
	{
		_runID = _idSeq.getNextID();
		
		ExternalFunctionInvocationInstruction inst = null;
		
		// execute package function
		for (int i=0; i < _inst.size(); i++) 
		{
			try {
				inst = (ExternalFunctionInvocationInstruction)_inst.get(i);
				executeInstruction( ec, inst );
			}
			catch (Exception e){
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating instruction " + i + " in external function programBlock. inst: " + inst.toString(), e);
			}
		}
		
		// check return values
		checkOutputParameters(ec.getVariables());
	}
	
	/**
	 * Executes the external function instruction.
	 * 
	 */
	@Override
	public void executeInstruction(ExecutionContext ec, ExternalFunctionInvocationInstruction inst) 
		throws DMLRuntimeException 
	{
		// After the udf framework rework, we moved the code of ExternalFunctionProgramBlockCP 
		// to ExternalFunctionProgramBlock and hence hence both types of external functions can
		// share the same code path here.
		super.executeInstruction(ec, inst);
	}
	

	@Override
	protected void createInstructions() 
	{
		_inst = new ArrayList<Instruction>();

		// assemble information provided through keyvalue pairs
		String className = _otherParams.get(ExternalFunctionStatement.CLASS_NAME);
		String configFile = _otherParams.get(ExternalFunctionStatement.CONFIG_FILE);
		
		// class name cannot be null, however, configFile and execLocation can be null
		if (className == null)
			throw new RuntimeException(this.printBlockErrorLocation() + ExternalFunctionStatement.CLASS_NAME + " not provided!");

		// assemble input and output param strings
		String inputParameterString = getParameterString(getInputParams());
		String outputParameterString = getParameterString(getOutputParams());

		// generate instruction
		ExternalFunctionInvocationInstruction einst = new ExternalFunctionInvocationInstruction(
				className, configFile, inputParameterString,
				outputParameterString);

		_inst.add(einst);

	}

	@Override
	protected void modifyInputMatrix(Matrix m, MatrixObject mobj) 
	{
		//pass in-memory object to external function
		m.setMatrixObject( mobj );
	}
	
	@Override
	protected MatrixObject createOutputMatrixObject(Matrix m)
	{
		MatrixObject ret = m.getMatrixObject();
		
		if( ret == null ) //otherwise, pass in-memory matrix from extfunct back to invoking program
		{
			MatrixCharacteristics mc = new MatrixCharacteristics(m.getNumRows(),m.getNumCols(), ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
			MatrixFormatMetaData mfmd = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			ret = new MatrixObject(ValueType.DOUBLE, m.getFilePath(), mfmd);
		}
		
		//for allowing in-memory packagesupport matrices w/o filesnames
		if( ret.getFileName().equals( DEFAULT_FILENAME ) ) 
		{
			ret.setFileName( createDefaultOutputFilePathAndName() );
		}
			
		return ret;
	}	
	
	
	public String createDefaultOutputFilePathAndName( )
	{
		return _baseDir + DEFAULT_FILENAME + _defaultSeq.getNextID();
	}	

	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in external function program block (for CP) generated from external function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
}