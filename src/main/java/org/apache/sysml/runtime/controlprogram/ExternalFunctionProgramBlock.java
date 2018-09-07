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
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.TreeMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.compile.JobType;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.ExternalFunctionStatement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.MRJobInstruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.VariableCPInstruction;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.ExternalFunctionInvocationInstruction;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar.ScalarValueType;

public class ExternalFunctionProgramBlock extends FunctionProgramBlock 
{
	protected static final IDSequence _idSeq = new IDSequence();

	protected long _runID = -1; //ID for block of statements
	protected String _baseDir = null;
	protected HashMap<String, String> _otherParams; // holds other key value parameters 

	private ArrayList<Instruction> block2CellInst; 
	private ArrayList<Instruction> cell2BlockInst; 
	private  HashMap<String, String> _unblockedFileNames;
	private HashMap<String, String> _blockedFileNames;

	
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

		_unblockedFileNames = new HashMap<>();
		_blockedFileNames = new HashMap<>();
	
		// generate instructions
		createInstructions();
	}
	
	private void changeTmpInput( long id, ExecutionContext ec )
	{
		ArrayList<DataIdentifier> inputParams = getInputParams();
		block2CellInst = getBlock2CellInstructions(inputParams, _unblockedFileNames);
		
		//post processing FUNCTION PATCH
		for( String var : _skipInReblock )
		{
			Data dat = ec.getVariable(var);
			if( dat instanceof MatrixObject )
				_unblockedFileNames.put(var, ((MatrixObject)dat).getFileName());
		}
	}
	
	/**
	 * It is necessary to change the local temporary files as only file handles are passed out
	 * by the external function program block.
	 * 
	 * 
	 * @param id this field does nothing
	 */
	private void changeTmpOutput( long id ) {
		ArrayList<DataIdentifier> outputParams = getOutputParams();
		cell2BlockInst = getCell2BlockInstructions(outputParams, _blockedFileNames);
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
		
		changeTmpInput( _runID, ec ); 
		changeTmpOutput( _runID );
		
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
		block2CellInst = getBlock2CellInstructions(getInputParams(),_unblockedFileNames);

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
		cell2BlockInst = getCell2BlockInstructions(getOutputParams(),_blockedFileNames);
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
	 * Method to generate a reblock job to convert the cell representation into block representation
	 * 
	 * @param outputParams list out output data identifiers
	 * @param blockedFileNames map of blocked file names
	 * @return list of instructions
	 */
	private ArrayList<Instruction> getCell2BlockInstructions(
			ArrayList<DataIdentifier> outputParams,
			HashMap<String, String> blockedFileNames) {
		
		ArrayList<Instruction> c2binst = null;
		
		//list of matrices that need to be reblocked
		ArrayList<DataIdentifier> matrices = new ArrayList<>();
		ArrayList<DataIdentifier> matricesNoReblock = new ArrayList<>();

		// identify outputs that are matrices
		for (int i = 0; i < outputParams.size(); i++) {
			if( outputParams.get(i).getDataType().isMatrix() ) {
				if( _skipOutReblock.contains(outputParams.get(i).getName()) )
					matricesNoReblock.add(outputParams.get(i));
				else
					matrices.add(outputParams.get(i));
			}
		}

		if( !matrices.isEmpty() )
		{
			c2binst = new ArrayList<>();
			MRJobInstruction reblkInst = new MRJobInstruction(JobType.REBLOCK);
			TreeMap<Integer, ArrayList<String>> MRJobLineNumbers = null;
			if(DMLScript.ENABLE_DEBUG_MODE) {
				MRJobLineNumbers = new TreeMap<>();
			}
			
			ArrayList<String> inLabels = new ArrayList<>();
			ArrayList<String> outLabels = new ArrayList<>();
			String[] outputs = new String[matrices.size()];
			byte[] resultIndex = new byte[matrices.size()];
			String reblock = "";
			String reblockStr = ""; //Keep a copy of a single MR reblock instruction
	
			String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
			
			try {
				// create a RBLK job that transforms each output matrix from cell to block
				for (int i = 0; i < matrices.size(); i++) {
					inLabels.add(matrices.get(i).getName());
					outLabels.add(matrices.get(i).getName() + "_extFnOutput");
					outputs[i] = scratchSpaceLoc +
							Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
							_otherParams.get(ExternalFunctionStatement.CLASS_NAME) + _runID + "_" + i + "Output";
					blockedFileNames.put(matrices.get(i).getName(), outputs[i]);
					resultIndex[i] = (byte) i; // (matrices.size()+i);
		
					if (i > 0)
						reblock += Lop.INSTRUCTION_DELIMITOR;
		
					reblock += "MR" + Lop.OPERAND_DELIMITOR + "rblk" + Lop.OPERAND_DELIMITOR + 
									i + Lop.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lop.VALUETYPE_PREFIX + matrices.get(i).getValueType() + Lop.OPERAND_DELIMITOR + 
									i + Lop.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lop.VALUETYPE_PREFIX + matrices.get(i).getValueType() + Lop.OPERAND_DELIMITOR + 
									ConfigurationManager.getBlocksize() + Lop.OPERAND_DELIMITOR + ConfigurationManager.getBlocksize() + Lop.OPERAND_DELIMITOR + "true";
					
					if(DMLScript.ENABLE_DEBUG_MODE) {
						//Create a copy of reblock instruction but as a single instruction (FOR DEBUGGER)
						reblockStr = "MR" + Lop.OPERAND_DELIMITOR + "rblk" + Lop.OPERAND_DELIMITOR + 
										i + Lop.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lop.VALUETYPE_PREFIX + matrices.get(i).getValueType() + Lop.OPERAND_DELIMITOR + 
										i + Lop.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lop.VALUETYPE_PREFIX + matrices.get(i).getValueType() + Lop.OPERAND_DELIMITOR + 
										ConfigurationManager.getBlocksize() + Lop.OPERAND_DELIMITOR + ConfigurationManager.getBlocksize()  + Lop.OPERAND_DELIMITOR + "true";					
						//Set MR reblock instruction line number (FOR DEBUGGER)
						if (!MRJobLineNumbers.containsKey(matrices.get(i).getBeginLine())) {
							MRJobLineNumbers.put(matrices.get(i).getBeginLine(), new ArrayList<String>()); 
						}
						MRJobLineNumbers.get(matrices.get(i).getBeginLine()).add(reblockStr);					
					}
					// create metadata instructions to populate symbol table 
					// with variables that hold blocked matrices
					Instruction createInst = VariableCPInstruction.prepareCreateMatrixVariableInstruction(outLabels.get(i), outputs[i], false, OutputInfo.outputInfoToString(OutputInfo.BinaryBlockOutputInfo));
					createInst.setLocation(matrices.get(i));
					
					c2binst.add(createInst);

				}
		
				reblkInst.setReBlockInstructions(inLabels.toArray(new String[inLabels.size()]), "", reblock, "", 
						outLabels.toArray(new String[inLabels.size()]), resultIndex, 1, 1);
				c2binst.add(reblkInst);
		
				// generate instructions that rename the output variables of REBLOCK job
				Instruction cpInst = null, rmInst = null;
				for (int i = 0; i < matrices.size(); i++) {
					cpInst = VariableCPInstruction.prepareCopyInstruction(outLabels.get(i), matrices.get(i).getName());
					rmInst = VariableCPInstruction.prepareRemoveInstruction(outLabels.get(i));
					
					cpInst.setLocation(matrices.get(i));
					rmInst.setLocation(matrices.get(i));
					
					c2binst.add(cpInst);
					c2binst.add(rmInst);
					//c2binst.add(CPInstructionParser.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "cpvar"+Lops.OPERAND_DELIMITOR+ outLabels.get(i) + Lops.OPERAND_DELIMITOR + matrices.get(i).getName()));
				}
			} catch (Exception e) {
				throw new RuntimeException(this.printBlockErrorLocation() + "error generating instructions", e);
			}
			
			//LOGGING instructions
			if (LOG.isTraceEnabled()){
				LOG.trace("\n--- Cell-2-Block Instructions ---");
				for(Instruction i : c2binst) {
					LOG.trace(i.toString());
				}
				LOG.trace("----------------------------------");
			}
			
		}
		
		return c2binst; //null if no output matrices
	}

	/**
	 * Method to generate instructions to convert input matrices from block to
	 * cell. We generate a GMR job here.
	 * 
	 * @param inputParams list of data identifiers
	 * @param unBlockedFileNames map of unblocked file names
	 * @return list of instructions
	 */
	private ArrayList<Instruction> getBlock2CellInstructions(
			ArrayList<DataIdentifier> inputParams,
			HashMap<String, String> unBlockedFileNames) {
		
		ArrayList<Instruction> b2cinst = null;
		
		//list of input matrices
		ArrayList<DataIdentifier> matrices = new ArrayList<>();
		ArrayList<DataIdentifier> matricesNoReblock = new ArrayList<>();

		// find all inputs that are matrices
		for (int i = 0; i < inputParams.size(); i++) {
			if( inputParams.get(i).getDataType().isMatrix() ) {
				if( _skipInReblock.contains(inputParams.get(i).getName()) )
					matricesNoReblock.add(inputParams.get(i));
				else
					matrices.add(inputParams.get(i));
			}
		}
		
		if( !matrices.isEmpty() )
		{
			b2cinst = new ArrayList<>();
			MRJobInstruction gmrInst = new MRJobInstruction(JobType.GMR);
			TreeMap<Integer, ArrayList<String>> MRJobLineNumbers = null;
			if(DMLScript.ENABLE_DEBUG_MODE) {
				MRJobLineNumbers = new TreeMap<>();
			}
			String gmrStr="";
			ArrayList<String> inLabels = new ArrayList<>();
			ArrayList<String> outLabels = new ArrayList<>();
			String[] outputs = new String[matrices.size()];
			byte[] resultIndex = new byte[matrices.size()];
	
			String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
			
			
			try {
				// create a GMR job that transforms each of these matrices from block to cell
				for (int i = 0; i < matrices.size(); i++) {
					
					inLabels.add(matrices.get(i).getName());
					outLabels.add(matrices.get(i).getName()+"_extFnInput");
					resultIndex[i] = (byte) i; //(matrices.size()+i);
	
					outputs[i] = scratchSpaceLoc +
									Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
									_otherParams.get(ExternalFunctionStatement.CLASS_NAME) + _runID + "_" + i + "Input";
					unBlockedFileNames.put(matrices.get(i).getName(), outputs[i]);
	
					if(DMLScript.ENABLE_DEBUG_MODE) {
						//Create a dummy gmr instruction (FOR DEBUGGER)
						gmrStr = "MR" + Lop.OPERAND_DELIMITOR + "gmr" + Lop.OPERAND_DELIMITOR + 
										i + Lop.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lop.VALUETYPE_PREFIX + matrices.get(i).getValueType() + Lop.OPERAND_DELIMITOR + 
										i + Lop.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lop.VALUETYPE_PREFIX + matrices.get(i).getValueType() + Lop.OPERAND_DELIMITOR + 
										ConfigurationManager.getBlocksize() + Lop.OPERAND_DELIMITOR + ConfigurationManager.getBlocksize();
						
						//Set MR gmr instruction line number (FOR DEBUGGER)
						if (!MRJobLineNumbers.containsKey(matrices.get(i).getBeginLine())) {
							MRJobLineNumbers.put(matrices.get(i).getBeginLine(), new ArrayList<String>()); 
						}
						MRJobLineNumbers.get(matrices.get(i).getBeginLine()).add(gmrStr);
					}
					// create metadata instructions to populate symbol table 
					// with variables that hold unblocked matrices
				 	Instruction createInst = VariableCPInstruction.prepareCreateMatrixVariableInstruction(outLabels.get(i), outputs[i], false, OutputInfo.outputInfoToString(OutputInfo.TextCellOutputInfo));			 		
			 		createInst.setLocation(matrices.get(i));
			 		
			 		b2cinst.add(createInst);
				}
			
				// Finally, generate GMR instruction that performs block2cell conversion
				gmrInst.setGMRInstructions(inLabels.toArray(new String[inLabels.size()]), "", "", "", "", 
						outLabels.toArray(new String[outLabels.size()]), resultIndex, 0, 1);
				
				b2cinst.add(gmrInst);
			
				// generate instructions that rename the output variables of GMR job
				Instruction cpInst=null, rmInst=null;
				for (int i = 0; i < matrices.size(); i++) {
						cpInst = VariableCPInstruction.prepareCopyInstruction(outLabels.get(i), matrices.get(i).getName());
						rmInst = VariableCPInstruction.prepareRemoveInstruction(outLabels.get(i));
						
						cpInst.setLocation(matrices.get(i));
						rmInst.setLocation(matrices.get(i));
						
						b2cinst.add(cpInst);
						b2cinst.add(rmInst);
				}
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		
			//LOG instructions
			if (LOG.isTraceEnabled()){
				LOG.trace("\n--- Block-2-Cell Instructions ---");
				for(Instruction i : b2cinst) {
					LOG.trace(i.toString());
				}
				LOG.trace("----------------------------------");
			}			
		}
		
		//BEGIN FUNCTION PATCH
		if( !matricesNoReblock.isEmpty() )
		{	
			for( int i=0; i<matricesNoReblock.size(); i++ )
			{
				String scratchSpaceLoc = ConfigurationManager.getScratchSpace();
				String filename = scratchSpaceLoc +
							          Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
							           _otherParams.get(ExternalFunctionStatement.CLASS_NAME) + _runID + "_" + i + "Input";
				unBlockedFileNames.put(matricesNoReblock.get(i).getName(), filename); 			
			}
		}
		//END FUNCTION PATCH
		
		return b2cinst; //null if no input matrices
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
	
	
	/////////////////////////////////////////////////
	// Extension for Global Data Flow Optimization
	// by Mathias Peters
	///////
	
	//FUNCTION PATCH
	
	private Collection<String> _skipInReblock = new HashSet<>();
	private Collection<String> _skipOutReblock = new HashSet<>();
	
	@Override
	public ArrayList<Instruction> getInstructions()
	{
		ArrayList<Instruction> tmp = new ArrayList<>();
		if( cell2BlockInst != null )
			tmp.addAll(cell2BlockInst);
		if( block2CellInst != null )
			tmp.addAll(block2CellInst);
		return tmp;
	}
}