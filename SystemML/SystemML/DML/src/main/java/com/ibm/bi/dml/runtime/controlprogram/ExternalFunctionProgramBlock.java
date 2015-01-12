/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Vector;

import org.nimble.configuration.NimbleConfig;
import org.nimble.control.DAGQueue;
import org.nimble.control.PMLDriver;
import org.nimble.exception.NimbleCheckedRuntimeException;
import org.nimble.task.AbstractTask;
import org.w3c.dom.Element;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.ExternalFunctionStatement;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.CacheException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.VariableCPInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.udf.ExternalFunctionInvocationInstruction;
import com.ibm.bi.dml.udf.FunctionParameter;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;
import com.ibm.bi.dml.udf.Scalar;
import com.ibm.bi.dml.udf.FunctionParameter.FunctionParameterType;
import com.ibm.bi.dml.udf.WrapperTaskForControlNode;
import com.ibm.bi.dml.udf.WrapperTaskForWorkerNode;
import com.ibm.bi.dml.udf.BinaryObject;
import com.ibm.bi.dml.udf.Scalar.ScalarValueType;

public class ExternalFunctionProgramBlock extends FunctionProgramBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	protected static IDSequence _idSeq = null;

	//handle to the nimble dag queue
	protected static DAGQueue _dagQueue = null;
	
	protected String _baseDir = null;

	ArrayList<Instruction> block2CellInst; 
	ArrayList<Instruction> cell2BlockInst; 

	// holds other key value parameters specified in function declaration
	protected HashMap<String, String> _otherParams;

	protected HashMap<String, String> _unblockedFileNames;
	protected HashMap<String, String> _blockedFileNames;

	protected long _runID = -1; //ID for block of statements
	
	static
	{
		_idSeq = new IDSequence();
	}
	
	/**
	 * Constructor that also provides otherParams that are needed for external
	 * functions. Remaining parameters will just be passed to constructor for
	 * function program block.
	 * 
	 * @param eFuncStat
	 * @throws DMLRuntimeException 
	 */
	protected ExternalFunctionProgramBlock(Program prog,
			Vector<DataIdentifier> inputParams,
			Vector<DataIdentifier> outputParams,
			String baseDir) throws DMLRuntimeException
	{
		super(prog, inputParams, outputParams);		
		_baseDir = baseDir;
		
		//NOTE: no need to setup nimble queue for CP external functions
	}
	
	public ExternalFunctionProgramBlock(Program prog,
			Vector<DataIdentifier> inputParams,
			Vector<DataIdentifier> outputParams,
			HashMap<String, String> otherParams,
			String baseDir) throws DMLRuntimeException {

		super(prog, inputParams, outputParams);
		_baseDir = baseDir;
		
		// copy other params
		_otherParams = new HashMap<String, String>();
		_otherParams.putAll(otherParams);

		_unblockedFileNames = new HashMap<String, String>();
		_blockedFileNames = new HashMap<String, String>();

		//setup nimble queue (if not existing)
		setupNIMBLEQueue();
		if (_dagQueue == null)
			LOG.warn("dagQueue is not set");
		
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
	 * @param id
	 */
	private void changeTmpOutput( long id )
	{
		ArrayList<DataIdentifier> outputParams = getOutputParams();
		cell2BlockInst = getCell2BlockInstructions(outputParams, _blockedFileNames);
	}
	
	/**
	 * 
	 * @return
	 */
	public String getBaseDir()
	{
		return _baseDir;
	}
	
	/**
	 * Method to be invoked to execute instructions for the external function
	 * invocation
	 * @throws DMLRuntimeException 
	 */
	@Override
	public void execute(ExecutionContext ec) 
		throws DMLRuntimeException
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
				if ( d.getDataType() == DataType.MATRIX ) {
					MatrixObject inputObj = (MatrixObject) d;
					inputObj.exportData();
				}
			}
		}
		catch (Exception e){
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Error exporting input variables to HDFS", e);
		}
		
		// convert block to cell
		if( block2CellInst != null )
		{
			ArrayList<Instruction> tempInst = new ArrayList<Instruction>();
			tempInst.addAll(block2CellInst);
			try {
				this.executeInstructions(tempInst,ec);
			} catch (Exception e) {
				
				throw new PackageRuntimeException(this.printBlockErrorLocation() + "Error executing "
						+ tempInst.toString(), e);
			}
		}
		
		// now execute package function
		for (int i = 0; i < _inst.size(); i++) {

			if (_inst.get(i) instanceof ExternalFunctionInvocationInstruction) {
				try {
					executeInstruction(ec,
							(ExternalFunctionInvocationInstruction) _inst.get(i),
							getDAGQueue());
				} catch (NimbleCheckedRuntimeException e) {
				
					throw new PackageRuntimeException(this.printBlockErrorLocation() + 
							"Failed to execute instruction "
									+ _inst.get(i).toString(), e);
				}
			}
		}

		// convert cell to block
		if( cell2BlockInst != null )
		{
			ArrayList<Instruction> tempInst = new ArrayList<Instruction>();
			try {
				tempInst.clear();
				tempInst.addAll(cell2BlockInst);
				this.executeInstructions(tempInst, ec);
			} catch (Exception e) {
				
				throw new PackageRuntimeException(this.printBlockErrorLocation() + "Failed to execute instruction "
						+ cell2BlockInst.toString(), e);
			}
		}
		
		// check return values
		checkOutputParameters(ec.getVariables());
	}

	/**
	 * Given a list of parameters as data identifiers, returns a string
	 * representation.
	 * 
	 * @param params
	 * @return
	 */

	protected String getParameterString(ArrayList<DataIdentifier> params) {
		String parameterString = "";

		for (int i = 0; i < params.size(); i++) {
			if (i != 0)
				parameterString += ",";

			DataIdentifier param = params.get(i);

			if (param.getDataType() == DataType.MATRIX) {
				String s = getDataTypeString(DataType.MATRIX) + ":";
				s = s + "" + param.getName() + "" + ":";
				s = s + getValueTypeString(param.getValueType());
				parameterString += s;
				continue;
			}

			if (param.getDataType() == DataType.SCALAR) {
				String s = getDataTypeString(DataType.SCALAR) + ":";
				s = s + "" + param.getName() + "" + ":";
				s = s + getValueTypeString(param.getValueType());
				parameterString += s;
				continue;
			}

			if (param.getDataType() == DataType.OBJECT) {
				String s = getDataTypeString(DataType.OBJECT) + ":";
				s = s + "" + param.getName() + "" + ":";
				parameterString += s;
				continue;
			}
		}

		return parameterString;
	}

	/**
	 * method to get instructions
	 */
	protected void createInstructions() {

		_inst = new ArrayList<Instruction>();

		// unblock all input matrices
		block2CellInst = getBlock2CellInstructions(getInputParams(),_unblockedFileNames);

		// assemble information provided through keyvalue pairs
		String className = _otherParams.get(ExternalFunctionStatement.CLASS_NAME);
		String configFile = _otherParams.get(ExternalFunctionStatement.CONFIG_FILE);
		String execLocation = _otherParams.get(ExternalFunctionStatement.EXEC_LOCATION);

		// class name cannot be null, however, configFile and execLocation can
		// be null
		if (className == null)
			throw new PackageRuntimeException(this.printBlockErrorLocation() + ExternalFunctionStatement.CLASS_NAME + " not provided!");

		// assemble input and output param strings
		String inputParameterString = getParameterString(getInputParams());
		String outputParameterString = getParameterString(getOutputParams());

		// generate instruction
		ExternalFunctionInvocationInstruction einst = new ExternalFunctionInvocationInstruction(
				className, configFile, execLocation, inputParameterString,
				outputParameterString);
		if(DMLScript.ENABLE_DEBUG_MODE) {
			if (getInputParams().size() > 0)
				einst.setLineNum(getInputParams().get(0).getBeginLine());
			else if (getOutputParams().size() > 0)
				einst.setLineNum(getOutputParams().get(0).getBeginLine());
			else
				einst.setLineNum(this._beginLine);
		}
		_inst.add(einst);

		// block output matrices
		cell2BlockInst = getCell2BlockInstructions(getOutputParams(),_blockedFileNames);
	}

	
	/**
	 * Method to generate a reblock job to convert the cell representation into block representation
	 * @param outputParams
	 * @param blockedFileNames
	 * @return
	 */
	private ArrayList<Instruction> getCell2BlockInstructions(
			ArrayList<DataIdentifier> outputParams,
			HashMap<String, String> blockedFileNames) {
		
		ArrayList<Instruction> c2binst = null;
		
		//list of matrices that need to be reblocked
		ArrayList<DataIdentifier> matrices = new ArrayList<DataIdentifier>();
		ArrayList<DataIdentifier> matricesNoReblock = new ArrayList<DataIdentifier>();

		// identify outputs that are matrices
		for (int i = 0; i < outputParams.size(); i++) {
			if (outputParams.get(i).getDataType() == DataType.MATRIX) {
				if( _skipOutReblock.contains(outputParams.get(i).getName()) )
					matricesNoReblock.add(outputParams.get(i));
				else
					matrices.add(outputParams.get(i));
			}
		}

		if( matrices.size() > 0 )
		{
			c2binst = new ArrayList<Instruction>();
			MRJobInstruction reblkInst = new MRJobInstruction(JobType.REBLOCK);
			TreeMap<Integer, ArrayList<String>> MRJobLineNumbers = null;
			if(DMLScript.ENABLE_DEBUG_MODE) {
				MRJobLineNumbers = new TreeMap<Integer, ArrayList<String>>();
			}
			
			ArrayList<String> inLabels = new ArrayList<String>();
			ArrayList<String> outLabels = new ArrayList<String>();
			String[] outputs = new String[matrices.size()];
			byte[] resultIndex = new byte[matrices.size()];
			String reblock = "";
			String reblockStr = ""; //Keep a copy of a single MR reblock instruction
	
			String scratchSpaceLoc = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
			
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
		
					reblock += "MR" + ReBlock.OPERAND_DELIMITOR + "rblk" + ReBlock.OPERAND_DELIMITOR + 
									i + ReBlock.DATATYPE_PREFIX + matrices.get(i).getDataType() + ReBlock.VALUETYPE_PREFIX + matrices.get(i).getValueType() + ReBlock.OPERAND_DELIMITOR + 
									i + ReBlock.DATATYPE_PREFIX + matrices.get(i).getDataType() + ReBlock.VALUETYPE_PREFIX + matrices.get(i).getValueType() + ReBlock.OPERAND_DELIMITOR + 
									DMLTranslator.DMLBlockSize + ReBlock.OPERAND_DELIMITOR + DMLTranslator.DMLBlockSize + ReBlock.OPERAND_DELIMITOR + "true";
					
					if(DMLScript.ENABLE_DEBUG_MODE) {
						//Create a copy of reblock instruction but as a single instruction (FOR DEBUGGER)
						reblockStr = "MR" + ReBlock.OPERAND_DELIMITOR + "rblk" + ReBlock.OPERAND_DELIMITOR + 
										i + ReBlock.DATATYPE_PREFIX + matrices.get(i).getDataType() + ReBlock.VALUETYPE_PREFIX + matrices.get(i).getValueType() + ReBlock.OPERAND_DELIMITOR + 
										i + ReBlock.DATATYPE_PREFIX + matrices.get(i).getDataType() + ReBlock.VALUETYPE_PREFIX + matrices.get(i).getValueType() + ReBlock.OPERAND_DELIMITOR + 
										DMLTranslator.DMLBlockSize + ReBlock.OPERAND_DELIMITOR + DMLTranslator.DMLBlockSize  + ReBlock.OPERAND_DELIMITOR + "true";					
						//Set MR reblock instruction line number (FOR DEBUGGER)
						if (!MRJobLineNumbers.containsKey(matrices.get(i).getBeginLine())) {
							MRJobLineNumbers.put(matrices.get(i).getBeginLine(), new ArrayList<String>()); 
						}
						MRJobLineNumbers.get(matrices.get(i).getBeginLine()).add(reblockStr);					
					}
					// create metadata instructions to populate symbol table 
					// with variables that hold blocked matrices
					
			  		/*StringBuilder mtdInst = new StringBuilder();
					mtdInst.append("CP" + Lops.OPERAND_DELIMITOR + "createvar");
			 		mtdInst.append(Lops.OPERAND_DELIMITOR + outLabels.get(i) + Lops.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lops.VALUETYPE_PREFIX + matrices.get(i).getValueType());
			  		mtdInst.append(Lops.OPERAND_DELIMITOR + outputs[i] + Lops.DATATYPE_PREFIX + DataType.SCALAR + Lops.VALUETYPE_PREFIX + ValueType.STRING);
			  		mtdInst.append(Lops.OPERAND_DELIMITOR + OutputInfo.outputInfoToString(OutputInfo.BinaryBlockOutputInfo) ) ;
					c2binst.add(CPInstructionParser.parseSingleInstruction(mtdInst.toString()));*/
					Instruction createInst = VariableCPInstruction.prepareCreateVariableInstruction(outLabels.get(i), outputs[i], false, OutputInfo.outputInfoToString(OutputInfo.BinaryBlockOutputInfo));
					if(DMLScript.ENABLE_DEBUG_MODE) {
						createInst.setLineNum(matrices.get(i).getBeginLine());
					}
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
					if(DMLScript.ENABLE_DEBUG_MODE) {
						cpInst.setLineNum(matrices.get(i).getBeginLine());
						rmInst.setLineNum(matrices.get(i).getBeginLine());
					}
					c2binst.add(cpInst);
					c2binst.add(rmInst);
					//c2binst.add(CPInstructionParser.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "cpvar"+Lops.OPERAND_DELIMITOR+ outLabels.get(i) + Lops.OPERAND_DELIMITOR + matrices.get(i).getName()));
				}
			} catch (Exception e) {
				throw new PackageRuntimeException(this.printBlockErrorLocation() + "error generating instructions", e);
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
	 * @param inputParams
	 * @return
	 */
	private ArrayList<Instruction> getBlock2CellInstructions(
			ArrayList<DataIdentifier> inputParams,
			HashMap<String, String> unBlockedFileNames) {
		
		ArrayList<Instruction> b2cinst = null;
		
		//list of input matrices
		ArrayList<DataIdentifier> matrices = new ArrayList<DataIdentifier>();
		ArrayList<DataIdentifier> matricesNoReblock = new ArrayList<DataIdentifier>();

		// find all inputs that are matrices
		for (int i = 0; i < inputParams.size(); i++) {
			if (inputParams.get(i).getDataType() == DataType.MATRIX) {
				if( _skipInReblock.contains(inputParams.get(i).getName()) )
					matricesNoReblock.add(inputParams.get(i));
				else
					matrices.add(inputParams.get(i));
			}
		}
		
		if( matrices.size()>0 )
		{
			b2cinst = new ArrayList<Instruction>();
			MRJobInstruction gmrInst = new MRJobInstruction(JobType.GMR);
			TreeMap<Integer, ArrayList<String>> MRJobLineNumbers = null;
			if(DMLScript.ENABLE_DEBUG_MODE) {
				MRJobLineNumbers = new TreeMap<Integer, ArrayList<String>>();
			}
			String gmrStr="";
			ArrayList<String> inLabels = new ArrayList<String>();
			ArrayList<String> outLabels = new ArrayList<String>();
			String[] outputs = new String[matrices.size()];
			byte[] resultIndex = new byte[matrices.size()];
	
			String scratchSpaceLoc = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
			
			
			try {
				// create a GMR job that transforms each of these matrices from block to cell
				for (int i = 0; i < matrices.size(); i++) {
					
					//inputs[i] = "##" + matrices.get(i).getName() + "##";
					//inputInfo[i] = binBlockInputInfo;
					//outputInfo[i] = textCellOutputInfo;
					//numRows[i] = numCols[i] = numRowsPerBlock[i] = numColsPerBlock[i] = -1;
					//resultDimsUnknown[i] = 1;
	
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
										DMLTranslator.DMLBlockSize + Lop.OPERAND_DELIMITOR + DMLTranslator.DMLBlockSize;
						
						//Set MR gmr instruction line number (FOR DEBUGGER)
						if (!MRJobLineNumbers.containsKey(matrices.get(i).getBeginLine())) {
							MRJobLineNumbers.put(matrices.get(i).getBeginLine(), new ArrayList<String>()); 
						}
						MRJobLineNumbers.get(matrices.get(i).getBeginLine()).add(gmrStr);
					}
					// create metadata instructions to populate symbol table 
					// with variables that hold unblocked matrices
				 	
					/*StringBuilder mtdInst = new StringBuilder();
					mtdInst.append("CP" + Lops.OPERAND_DELIMITOR + "createvar");
						mtdInst.append(Lops.OPERAND_DELIMITOR + outLabels.get(i) + Lops.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lops.VALUETYPE_PREFIX + matrices.get(i).getValueType());
				 		mtdInst.append(Lops.OPERAND_DELIMITOR + outputs[i] + Lops.DATATYPE_PREFIX + DataType.SCALAR + Lops.VALUETYPE_PREFIX + ValueType.STRING);
				 		mtdInst.append(Lops.OPERAND_DELIMITOR + OutputInfo.outputInfoToString(OutputInfo.TextCellOutputInfo) ) ;
					b2cinst.add(CPInstructionParser.parseSingleInstruction(mtdInst.toString()));*/
					
			 		Instruction createInst = VariableCPInstruction.prepareCreateVariableInstruction(outLabels.get(i), outputs[i], false, OutputInfo.outputInfoToString(OutputInfo.TextCellOutputInfo));
			 		if(DMLScript.ENABLE_DEBUG_MODE) {
			 			createInst.setLineNum(matrices.get(i).getBeginLine());
			 		}
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
						if(DMLScript.ENABLE_DEBUG_MODE) {
							cpInst.setLineNum(matrices.get(i).getBeginLine());
							rmInst.setLineNum(matrices.get(i).getBeginLine());
						}
						b2cinst.add(cpInst);
						b2cinst.add(rmInst);
				}
			} catch (Exception e) {
				throw new PackageRuntimeException(e);
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
		if( matricesNoReblock.size() > 0 )
		{
			//if( b2cinst==null )
			//	b2cinst = new ArrayList<Instruction>();
			
			for( int i=0; i<matricesNoReblock.size(); i++ )
			{
				String scratchSpaceLoc = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE);
				
				try{
					String filename = scratchSpaceLoc +
							          Lop.FILE_SEPARATOR + Lop.PROCESS_PREFIX + DMLScript.getUUID() + Lop.FILE_SEPARATOR + 
							           _otherParams.get(ExternalFunctionStatement.CLASS_NAME) + _runID + "_" + i + "Input";
					//String outLabel = matricesNoReblock.get(i).getName()+"_extFnInput";
					//Instruction createInst = VariableCPInstruction.prepareCreateVariableInstruction(outLabel, filename, false, OutputInfo.outputInfoToString(OutputInfo.TextCellOutputInfo));
					//Instruction cpInst = VariableCPInstruction.prepareCopyInstruction( matricesNoReblock.get(i).getName(), outLabel);
					
					unBlockedFileNames.put(matricesNoReblock.get(i).getName(), filename); //
					
					//b2cinst.add(createInst);
					//b2cinst.add(cpInst);
				}
				catch (Exception e) {
					throw new PackageRuntimeException(e);
				}
							
			}
		}
		//END FUNCTION PATCH
		
		return b2cinst; //null if no input matrices
	}

	/**
	 * Method to execute an external function invocation instruction.
	 * 
	 * @param inst
	 * @param dQueue
	 * @throws NimbleCheckedRuntimeException
	 * @throws DMLRuntimeException 
	 */

	public void executeInstruction(ExecutionContext ec, ExternalFunctionInvocationInstruction inst,
			DAGQueue dQueue) throws NimbleCheckedRuntimeException, DMLRuntimeException {

		String className = inst.getClassName();
		String configFile = inst.getConfigFile();

		if (className == null)
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Class name can't be null");

		// create instance of package function.

		Object o;
		try {
			o = Class.forName(className).newInstance();
		} catch (Exception e) {
			throw new PackageRuntimeException(this.printBlockErrorLocation() +
					"Error generating package function object " + e.toString());
		}

		if (!(o instanceof PackageFunction))
			throw new PackageRuntimeException(this.printBlockErrorLocation() + 
					"Class is not of type PackageFunction");

		PackageFunction func = (PackageFunction) o;

		// add inputs to this package function based on input parameter
		// and their mappings.
		setupInputs(func, inst.getInputParams(), ec.getVariables());
		func.setConfiguration(configFile);
		func.setBaseDir(_baseDir);

		AbstractTask t = null;

		// determine exec location, default is control node
		// and allocate the appropriate NIMBLE task
		if (inst.getExecLocation().equals(ExternalFunctionStatement.WORKER))
			t = new WrapperTaskForWorkerNode(func);
		else
			t = new WrapperTaskForControlNode(func);

		// execute task and wait for completion
		dQueue.pushTask(t);
		try {
			t = dQueue.waitOnTask(t);
		} catch (Exception e) {
			throw new PackageRuntimeException(e);
		}

		// get updated function
		PackageFunction returnFunc;
		if (inst.getExecLocation().equals(ExternalFunctionStatement.WORKER))
			returnFunc = ((WrapperTaskForWorkerNode) t)
			.getUpdatedPackageFunction();
		else
			returnFunc = ((WrapperTaskForControlNode) t)
			.getUpdatedPackageFunction();

		// verify output of function execution matches declaration
		// and add outputs to variableMapping and Metadata
		verifyAndAttachOutputs(ec, returnFunc, inst.getOutputParams());
	}

	/**
	 * Method to verify that function outputs match with declared outputs
	 * 
	 * @param returnFunc
	 * @param outputParams
	 * @throws DMLRuntimeException 
	 */
	protected void verifyAndAttachOutputs(ExecutionContext ec, PackageFunction returnFunc,
			String outputParams) throws DMLRuntimeException {

		ArrayList<String> outputs = getParameters(outputParams);
		// make sure they are of equal size first

		if (outputs.size() != returnFunc.getNumFunctionOutputs()) {
			throw new PackageRuntimeException(
					"Number of function outputs ("+returnFunc.getNumFunctionOutputs()+") " +
					"does not match with declaration ("+outputs.size()+").");
		}

		// iterate over each output and verify that type matches
		for (int i = 0; i < outputs.size(); i++) {
			StringTokenizer tk = new StringTokenizer(outputs.get(i), ":");
			ArrayList<String> tokens = new ArrayList<String>();
			while (tk.hasMoreTokens()) {
				tokens.add(tk.nextToken());
			}

			if (returnFunc.getFunctionOutput(i).getType() == FunctionParameterType.Matrix) {
				Matrix m = (Matrix) returnFunc.getFunctionOutput(i);

				if (!(tokens.get(0)
						.compareTo(getFunctionParameterDataTypeString(FunctionParameterType.Matrix)) == 0)
						|| !(tokens.get(2).compareTo(
								getMatrixValueTypeString(m.getValueType())) == 0)) {
					throw new PackageRuntimeException(
							"Function output '"+outputs.get(i)+"' does not match with declaration.");
				}

				// add result to variableMapping
				String varName = tokens.get(1);
				MatrixObject newVar = createOutputMatrixObject( m ); 
				newVar.setVarName(varName);
				
				/* cleanup not required because done at central position (FunctionCallCPInstruction)
				MatrixObjectNew oldVar = (MatrixObjectNew)getVariable(varName);
				if( oldVar!=null )
					oldVar.clearData();*/
				
				//getVariables().put(varName, newVar); //put/override in local symbol table
				ec.setVariable(varName, newVar);
				
				continue;
			}

			if (returnFunc.getFunctionOutput(i).getType() == FunctionParameterType.Scalar) {
				Scalar s = (Scalar) returnFunc.getFunctionOutput(i);

				if (!tokens.get(0).equals(getFunctionParameterDataTypeString(FunctionParameterType.Scalar))
						|| !tokens.get(2).equals(
								getScalarValueTypeString(s.getScalarType()))) {
					throw new PackageRuntimeException(
							"Function output '"+outputs.get(i)+"' does not match with declaration.");
				}

				// allocate and set appropriate object based on type
				ScalarObject scalarObject = null;
				ScalarValueType type = s.getScalarType();
				switch (type) {
				case Integer:
					scalarObject = new IntObject(tokens.get(1),
							Long.parseLong(s.getValue()));
					break;
				case Double:
					scalarObject = new DoubleObject(tokens.get(1),
							Double.parseDouble(s.getValue()));
					break;
				case Boolean:
					scalarObject = new BooleanObject(tokens.get(1),
							Boolean.parseBoolean(s.getValue()));
					break;
				case Text:
					scalarObject = new StringObject(tokens.get(1), s.getValue());
					break;
				default:
					throw new PackageRuntimeException(
							"Unknown scalar value type '"+type+"' of output '"+outputs.get(i)+"'.");
				}

				//this.getVariables().put(tokens.get(1), scalarObject);
				ec.setVariable(tokens.get(1), scalarObject);
				continue;
			}

			if (returnFunc.getFunctionOutput(i).getType() == FunctionParameterType.Object) {
				if (!tokens.get(0).equals(getFunctionParameterDataTypeString(FunctionParameterType.Object))) {
					throw new PackageRuntimeException(
							"Function output '"+outputs.get(i)+"' does not match with declaration.");
				}

				throw new PackageRuntimeException(
						"Object types not yet supported");

				// continue;
			}

			throw new PackageRuntimeException(
					"Unknown data type '"+returnFunc.getFunctionOutput(i).getType()+"' " +
					"of output '"+outputs.get(i)+"'.");
		}
	}

	protected MatrixObject createOutputMatrixObject( Matrix m ) 
		throws CacheException 
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(m.getNumRows(),m.getNumCols(), DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
		MatrixFormatMetaData mfmd = new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo);		
		return new MatrixObject(ValueType.DOUBLE, m.getFilePath(), mfmd);
	}

	/**
	 * Method to get string representation of scalar value type
	 * 
	 * @param scalarType
	 * @return
	 */

	protected String getScalarValueTypeString(ScalarValueType scalarType) {

		if (scalarType.equals(ScalarValueType.Double))
			return "Double";
		if (scalarType.equals(ScalarValueType.Integer))
			return "Integer";
		if (scalarType.equals(ScalarValueType.Boolean))
			return "Boolean";
		if (scalarType.equals(ScalarValueType.Text))
			return "String";

		throw new PackageRuntimeException("Unknown scalar value type");
	}

	/**
	 * Method to parse inputs, update labels, and add to package function.
	 * 
	 * @param func
	 * @param inputParams
	 * @param metaData
	 * @param variableMapping
	 */
	protected void setupInputs (PackageFunction func, String inputParams,
			LocalVariableMap variableMapping) {

		ArrayList<String> inputs = getParameters(inputParams);
		ArrayList<FunctionParameter> inputObjects = getInputObjects(inputs, variableMapping);
		func.setNumFunctionInputs(inputObjects.size());
		for (int i = 0; i < inputObjects.size(); i++)
			func.setInput(inputObjects.get(i), i);

	}

	/**
	 * Method to convert string representation of input into function input
	 * object.
	 * 
	 * @param inputs
	 * @param variableMapping
	 * @param metaData
	 * @return
	 */

	protected ArrayList<FunctionParameter> getInputObjects(ArrayList<String> inputs,
			LocalVariableMap variableMapping) {
		ArrayList<FunctionParameter> inputObjects = new ArrayList<FunctionParameter>();

		for (int i = 0; i < inputs.size(); i++) {
			ArrayList<String> tokens = new ArrayList<String>();
			StringTokenizer tk = new StringTokenizer(inputs.get(i), ":");
			while (tk.hasMoreTokens()) {
				tokens.add(tk.nextToken());
			}

			if (tokens.get(0).equals("Matrix")) {
				String varName = tokens.get(1);
				MatrixObject mobj = (MatrixObject) variableMapping.get(varName);
				MatrixDimensionsMetaData md = (MatrixDimensionsMetaData) mobj.getMetaData();
				Matrix m = new Matrix(mobj.getFileName(),
						md.getMatrixCharacteristics().numRows,
						md.getMatrixCharacteristics().numColumns,
						getMatrixValueType(tokens.get(2)));
				modifyInputMatrix(m, mobj);
				inputObjects.add(m);
			}

			if (tokens.get(0).equals("Scalar")) {
				String varName = tokens.get(1);
				ScalarObject so = (ScalarObject) variableMapping.get(varName);
				Scalar s = new Scalar(getScalarValueType(tokens.get(2)),
						so.getStringValue());
				inputObjects.add(s);

			}

			if (tokens.get(0).equals("Object")) {
				String varName = tokens.get(1);
				Object o = variableMapping.get(varName);
				BinaryObject obj = new BinaryObject(o);
				inputObjects.add(obj);

			}
		}

		return inputObjects;

	}

	protected void modifyInputMatrix(Matrix m, MatrixObject mobj) 
	{
		//do nothing, intended for extensions
	}

	/**
	 * Converts string representation of scalar value type to enum type
	 * 
	 * @param string
	 * @return
	 */
	protected ScalarValueType getScalarValueType(String string) {
		if (string.equals("Double"))
			return ScalarValueType.Double;
		if (string.equals("Integer"))
			return ScalarValueType.Integer;
		if (string.equals("Boolean"))
			return ScalarValueType.Boolean;
		if (string.equals("String"))
			return ScalarValueType.Text;

		throw new PackageRuntimeException("Unknown scalar type");

	}

	/**
	 * Get string representation of matrix value type
	 * 
	 * @param t
	 * @return
	 */

	protected String getMatrixValueTypeString(Matrix.ValueType t) {
		if (t.equals(Matrix.ValueType.Double))
			return "Double";

		if (t.equals(Matrix.ValueType.Integer))
			return "Integer";

		throw new PackageRuntimeException("Unknown matrix value type");
	}

	/**
	 * Converts string representation of matrix value type into enum type
	 * 
	 * @param string
	 * @return
	 */

	protected com.ibm.bi.dml.udf.Matrix.ValueType getMatrixValueType(String string) {

		if (string.equals("Double"))
			return Matrix.ValueType.Double;
		if (string.equals("Integer"))
			return Matrix.ValueType.Integer;

		throw new PackageRuntimeException("Unknown matrix value type");

	}

	/**
	 * Method to break the comma separated input parameters into an arraylist of
	 * parameters
	 * 
	 * @param inputParams
	 * @return
	 */
	protected ArrayList<String> getParameters(String inputParams) {
		ArrayList<String> inputs = new ArrayList<String>();

		StringTokenizer tk = new StringTokenizer(inputParams, ",");
		while (tk.hasMoreTokens()) {
			inputs.add(tk.nextToken());
		}

		return inputs;
	}

	/**
	 * Get string representation for data type
	 * 
	 * @param d
	 * @return
	 */
	protected String getDataTypeString(DataType d) {
		if (d.equals(DataType.MATRIX))
			return "Matrix";

		if (d.equals(DataType.SCALAR))
			return "Scalar";

		if (d.equals(DataType.OBJECT))
			return "Object";

		throw new PackageRuntimeException("Should never come here");

	}

	/**
	 * Method to get string representation of data type.
	 * 
	 * @param t
	 * @return
	 */
	protected String getFunctionParameterDataTypeString(FunctionParameterType t) {
		if (t.equals(FunctionParameterType.Matrix))
			return "Matrix";

		if (t.equals(FunctionParameterType.Scalar))
			return "Scalar";

		if (t.equals(FunctionParameterType.Object))
			return "Object";

		throw new PackageRuntimeException("Should never come here");
	}

	/**
	 * Get string representation of value type
	 * 
	 * @param v
	 * @return
	 */
	protected String getValueTypeString(ValueType v) {
		if (v.equals(ValueType.DOUBLE))
			return "Double";

		if (v.equals(ValueType.INT))
			return "Integer";

		if (v.equals(ValueType.BOOLEAN))
			return "Boolean";

		if (v.equals(ValueType.STRING))
			return "String";

		throw new PackageRuntimeException("Should never come here");
	}

	public void printMe() {
		//System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
	
	public HashMap<String,String> getOtherParams()
	{
		return _otherParams;
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in external function program block generated from external function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
	
	/**
	 * Method to setup the NIMBLE task queue. 
	 * This will be used in future external function invocations
	 * @param dmlCfg DMLConfig object
	 * @return NIMBLE task queue
	 */
	public synchronized static void setupNIMBLEQueue() 
	{
		if( _dagQueue == null )
		{
			DMLConfig dmlCfg = ConfigurationManager.getConfig();	
			
			//config not provided
			if (dmlCfg == null) 
				return;
			
			// read in configuration files
			NimbleConfig config = new NimbleConfig();
	
			try {
				config.parseSystemDocuments(dmlCfg.getConfig_file_name());
				
				//ensure unique working directory for nimble output
				StringBuffer sb = new StringBuffer();
				sb.append( dmlCfg.getTextValue(DMLConfig.SCRATCH_SPACE) );
				sb.append( Lop.FILE_SEPARATOR );
				sb.append( Lop.PROCESS_PREFIX );
				sb.append( DMLScript.getUUID() );
				sb.append( Lop.FILE_SEPARATOR  );
				sb.append( dmlCfg.getTextValue(DMLConfig.NIMBLE_SCRATCH) );			
				((Element)config.getSystemConfig().getParameters().getElementsByTagName(DMLConfig.NIMBLE_SCRATCH).item(0))
				                .setTextContent( sb.toString() );						
			} catch (Exception e) {
				throw new PackageRuntimeException ("Error parsing Nimble configuration files", e);
			}
	
			// get threads configuration and validate
			int numSowThreads = 1;
			int numReapThreads = 1;
	
			numSowThreads = Integer.parseInt
					(NimbleConfig.getTextValue(config.getSystemConfig().getParameters(), DMLConfig.NUM_SOW_THREADS));
			numReapThreads = Integer.parseInt
					(NimbleConfig.getTextValue(config.getSystemConfig().getParameters(), DMLConfig.NUM_REAP_THREADS));
			
			if (numSowThreads < 1 || numReapThreads < 1){
				throw new PackageRuntimeException("Illegal values for thread count (must be > 0)");
			}
	
			// Initialize an instance of the driver.
			PMLDriver driver = null;
			try {
				driver = new PMLDriver(numSowThreads, numReapThreads, config);
				driver.startEmptyDriver(config);
			} catch (Exception e) {
				throw new PackageRuntimeException("Problem starting nimble driver", e);
			} 
	
			_dagQueue = driver.getDAGQueue();
		}
	}

	public static DAGQueue getDAGQueue()
	{
		return _dagQueue;
	}
	
	public synchronized static void shutDownNimbleQueue()
	{
		//cleanup all nimble threads
		if(_dagQueue != null)
	  	    _dagQueue.forceShutDown();
		_dagQueue = null;
	}
	
	
	/////////////////////////////////////////////////
	// Extension for Global Data Flow Optimization
	// by Mathias Peters
	///////
	
	//FUNCTION PATCH
	
	private Collection<String> _skipInReblock = new HashSet<String>();
	private Collection<String> _skipOutReblock = new HashSet<String>();
	
	public void setSkippedReblockLists( Collection<String> varsIn, Collection<String> varsOut )
	{
		_skipInReblock.clear();
		_skipOutReblock.clear();
		
		if( varsIn!=null || varsOut!=null )
		{
			if( varsIn != null )
				_skipInReblock.addAll(varsIn);		
			if( varsOut != null )
				_skipOutReblock.addAll(varsOut);
		
			 //regenerate instructions
			createInstructions();
		}
	}
	
	
	@Override
	public ArrayList<Instruction> getInstructions()
	{
		ArrayList<Instruction> tmp = new ArrayList<Instruction>();
		if( cell2BlockInst != null )
			tmp.addAll(cell2BlockInst);
		if( block2CellInst != null )
			tmp.addAll(block2CellInst);
		return tmp;
	}
}