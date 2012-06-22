package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.StringTokenizer;
import java.util.Vector;

import org.nimble.control.DAGQueue;
import org.nimble.exception.NimbleCheckedRuntimeException;
import org.nimble.task.AbstractTask;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.compile.JobType;
import com.ibm.bi.dml.packagesupport.ExternalFunctionInvocationInstruction;
import com.ibm.bi.dml.packagesupport.FIO;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
import com.ibm.bi.dml.packagesupport.Scalar;
import com.ibm.bi.dml.packagesupport.Type;
import com.ibm.bi.dml.packagesupport.WrapperTaskForControlNode;
import com.ibm.bi.dml.packagesupport.WrapperTaskForWorkerNode;
import com.ibm.bi.dml.packagesupport.bObject;
import com.ibm.bi.dml.packagesupport.Scalar.ScalarType;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.CPInstructionParser;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixDimensionsMetaData;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.CacheOutOfMemoryException;
import com.ibm.bi.dml.utils.CacheStatusException;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.configuration.DMLConfig;

public class ExternalFunctionProgramBlock extends FunctionProgramBlock {

	protected static IDSequence _idSeq;

	public static final String CLASSNAME = "classname";
	public static final String EXECLOCATION = "execlocation";
	public static final String CONFIGFILE = "configfile";
	final String WORKER = "worker";
	final int ROWS_PER_BLOCK = DMLTranslator.DMLBlockSize;
	final int COLS_PER_BLOCK = DMLTranslator.DMLBlockSize;

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
			Vector<DataIdentifier> outputParams) throws DMLRuntimeException
	{
		super(prog, inputParams, outputParams);
	}
	
	public ExternalFunctionProgramBlock(Program prog,
			Vector<DataIdentifier> inputParams,
			Vector<DataIdentifier> outputParams,
			HashMap<String, String> otherParams) throws DMLRuntimeException {

		super(prog, inputParams, outputParams);

		// copy other params
		_otherParams = new HashMap<String, String>();
		_otherParams.putAll(otherParams);

		_unblockedFileNames = new HashMap<String, String>();
		_blockedFileNames = new HashMap<String, String>();

		// generate instructions
		createInstructions();
	}
	
	private void changeTmpInput( long id )
	{
		ArrayList<DataIdentifier> inputParams = getInputParams();
		block2CellInst = getBlock2CellInstructions(inputParams, _unblockedFileNames);
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
	 * Method to be invoked to execute instructions for the external function
	 * invocation
	 * @throws DMLRuntimeException 
	 */

	public void execute(ExecutionContext ec) throws DMLRuntimeException {
	
		_runID = _idSeq.getNextID();
		
		changeTmpInput( _runID ); 
		changeTmpOutput( _runID );
				
		// convert block to cell
		ArrayList<Instruction> tempInst = new ArrayList<Instruction>();
		tempInst.addAll(block2CellInst);
		try {
			this.execute(tempInst,ec);
		} catch (Exception e) {
			e.printStackTrace();
			throw new PackageRuntimeException("Error executing "
					+ tempInst.toString());
		}

		// now execute package function
		for (int i = 0; i < _inst.size(); i++) {

			if (_inst.get(i) instanceof ExternalFunctionInvocationInstruction) {
				try {
					executeInstruction(
							(ExternalFunctionInvocationInstruction) _inst
							.get(i),
							this._prog.getDAGQueue());
				} catch (NimbleCheckedRuntimeException e) {

					throw new PackageRuntimeException(
							"Failed to execute instruction "
									+ _inst.get(i).toString());
				}
			}
		}

		// convert cell to block
		try {
			tempInst.clear();
			tempInst.addAll(cell2BlockInst);
			this.execute(tempInst, ec);
		} catch (Exception e) {
			e.printStackTrace();
			throw new PackageRuntimeException("Failed to execute instruction "
					+ cell2BlockInst.toString());
		}
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
		block2CellInst = (getBlock2CellInstructions(getInputParams(),
				_unblockedFileNames));

		// assemble information provided through keyvalue pairs
		String className = _otherParams.get(CLASSNAME);
		String configFile = _otherParams.get(CONFIGFILE);
		String execLocation = _otherParams.get(EXECLOCATION);

		// class name cannot be null, however, configFile and execLocation can
		// be null
		if (className == null)
			throw new PackageRuntimeException(CLASSNAME + " not provided!");

		// assemble input and output param strings
		String inputParameterString = getParameterString(getInputParams());
		String outputParameterString = getParameterString(getOutputParams());

		// generate instruction
		ExternalFunctionInvocationInstruction einst = new ExternalFunctionInvocationInstruction(
				className, configFile, execLocation, inputParameterString,
				outputParameterString);

		_inst.add(einst);

		// block output matrices
		cell2BlockInst = getCell2BlockInstructions(getOutputParams(),
				_blockedFileNames);
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
		ArrayList<Instruction> c2binst = new ArrayList<Instruction>();

		//if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE ) {
		//	return c2binst;
		//}

		MRJobInstruction reblkInst = new MRJobInstruction(JobType.REBLOCK_BINARY);
		
		//list of matrices that need to be reblocked
		ArrayList<DataIdentifier> matrices = new ArrayList<DataIdentifier>();

		// identify outputs that are matrices
		for (int i = 0; i < outputParams.size(); i++) {
			if (outputParams.get(i).getDataType() == DataType.MATRIX) {
				matrices.add(outputParams.get(i));
			}
		}

		InputInfo textCellInputInfo = InputInfo.TextCellInputInfo;
		OutputInfo binaryBlockOutputInfo = OutputInfo.BinaryBlockOutputInfo;

		String[] inputs = new String[matrices.size()];
		String[] outputs = new String[matrices.size()];
		InputInfo[] inputInfo = new InputInfo[matrices.size()];
		OutputInfo[] outputInfo = new OutputInfo[matrices.size()];
		long[] numRows = new long[matrices.size()];
		long[] numCols = new long[matrices.size()];
		int[] numRowsPerBlock = new int[matrices.size()];
		int[] numColsPerBlock = new int[matrices.size()];
		byte[] resultIndex = new byte[matrices.size()];
		byte[] resultDimsUnknown = new byte[matrices.size()];
		ArrayList<String> inLabels = new ArrayList<String>();
		ArrayList<String> outLabels = new ArrayList<String>();
		String reblock = "";

		// create a RBLK job that transforms each of these matrices from cell to
		// block
		for (int i = 0; i < matrices.size(); i++) {

			inputs[i] = "##" + matrices.get(i).getName() + "##";
			outputs[i] = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE) + "/" +
                         _otherParams.get(CLASSNAME) + _runID + "_" + i + "Output";
			blockedFileNames.put(matrices.get(i).getName(), outputs[i]);
			inputInfo[i] = textCellInputInfo;
			outputInfo[i] = binaryBlockOutputInfo;
			inLabels.add(matrices.get(i).getName());
			outLabels.add(matrices.get(i).getName() + "_extFnOutput");
			numRows[i] = numCols[i] = numRowsPerBlock[i] = numColsPerBlock[i] = -1;
			resultIndex[i] = (byte) i;
			resultDimsUnknown[i] = 1;

			if (i > 0)
				reblock += ",";

			reblock += "MR" + ReBlock.OPERAND_DELIMITOR + "rblk" + ReBlock.OPERAND_DELIMITOR + i
					+ ReBlock.VALUETYPE_PREFIX + matrices.get(i).getValueType()
					+ ReBlock.OPERAND_DELIMITOR + i + ReBlock.VALUETYPE_PREFIX
					+ matrices.get(i).getValueType()
					+ ReBlock.OPERAND_DELIMITOR + ROWS_PER_BLOCK
					+ ReBlock.OPERAND_DELIMITOR + COLS_PER_BLOCK;
			
			// create metadata instructions to populate symbol table 
			// with variables that hold blocked matrices
	  		StringBuilder mtdInst = new StringBuilder();
			mtdInst.append("CP" + Lops.OPERAND_DELIMITOR + "createvar");
	 		mtdInst.append(Lops.OPERAND_DELIMITOR + outLabels.get(i) + Lops.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lops.VALUETYPE_PREFIX + matrices.get(i).getValueType());
	  		mtdInst.append(Lops.OPERAND_DELIMITOR + outputs[i] + Lops.DATATYPE_PREFIX + DataType.SCALAR + Lops.VALUETYPE_PREFIX + ValueType.STRING);
	  		mtdInst.append(Lops.OPERAND_DELIMITOR + OutputInfo.outputInfoToString(outputInfo[i]) ) ;
		
	  		try {
				c2binst.add(CPInstructionParser.parseSingleInstruction(mtdInst.toString()));
			} catch (Exception e) {
				throw new PackageRuntimeException(e);
			}
		}

		reblkInst.setReBlockInstructions(inputs, inputInfo, numRows, numCols,
				numRowsPerBlock, numColsPerBlock, "", reblock, "", outputs,
				outputInfo, resultIndex, resultDimsUnknown, 1, 1, inLabels,
				outLabels);
		c2binst.add(reblkInst);

		// generate instructions that rename the output variables of REBLOCK job
		for (int i = 0; i < matrices.size(); i++) {
			try {
				c2binst.add(CPInstructionParser.parseSingleInstruction("CP" + Lops.OPERAND_DELIMITOR + "mvvar"+Lops.OPERAND_DELIMITOR+ outLabels.get(i) + Lops.OPERAND_DELIMITOR + matrices.get(i).getName()));
			} catch (Exception e) {
				throw new PackageRuntimeException(e);
			}
		}
		
		//print instructions
		if (DMLScript.DEBUG) {
			System.out.println("--- Cell-2-Block Instructions ---");
			for(Instruction i : c2binst) {
				System.out.println(i.toString());
			}
		}
		
		return c2binst;
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
		ArrayList<Instruction> b2cinst = new ArrayList<Instruction>();
		
		//if ( DMLScript.rtplatform == RUNTIME_PLATFORM.SINGLE_NODE ) {
		//	return b2cinst;
		//}		
		
		MRJobInstruction gmrInst = new MRJobInstruction(JobType.GMR);

		//list of input matrices
		ArrayList<DataIdentifier> matrices = new ArrayList<DataIdentifier>();

		// find all inputs that are matrices

		for (int i = 0; i < inputParams.size(); i++) {
			if (inputParams.get(i).getDataType() == DataType.MATRIX) {
				matrices.add(inputParams.get(i));
			}
		}

		InputInfo binBlockInputInfo = InputInfo.BinaryBlockInputInfo;
		OutputInfo textCellOutputInfo = OutputInfo.TextCellOutputInfo;

		String[] inputs = new String[matrices.size()];
		String[] outputs = new String[matrices.size()];
		InputInfo[] inputInfo = new InputInfo[matrices.size()];
		OutputInfo[] outputInfo = new OutputInfo[matrices.size()];
		long[] numRows = new long[matrices.size()];
		long[] numCols = new long[matrices.size()];
		int[] numRowsPerBlock = new int[matrices.size()];
		int[] numColsPerBlock = new int[matrices.size()];
		byte[] resultIndex = new byte[matrices.size()];
		byte[] resultDimsUnknown = new byte[matrices.size()];
		ArrayList<String> inLabels = new ArrayList<String>();
		ArrayList<String> outLabels = new ArrayList<String>();

		// create a GMR job that transforms each of these matrices from block to
		// cell
		for (int i = 0; i < matrices.size(); i++) {
			inputs[i] = "##" + matrices.get(i).getName() + "##";
			outputs[i] = ConfigurationManager.getConfig().getTextValue(DMLConfig.SCRATCH_SPACE) + "/" +
                         _otherParams.get(CLASSNAME) + _runID + "_" + i + "Input";
			unBlockedFileNames.put(matrices.get(i).getName(), outputs[i]);
			inputInfo[i] = binBlockInputInfo;
			outputInfo[i] = textCellOutputInfo;
			inLabels.add(matrices.get(i).getName());
			outLabels.add(matrices.get(i).getName()+"_extFnInput");
			numRows[i] = numCols[i] = numRowsPerBlock[i] = numColsPerBlock[i] = -1;
			resultIndex[i] = (byte) i;
			resultDimsUnknown[i] = 1;
		
			// create metadata instructions to populate symbol table 
			// with variables that hold unblocked matrices
		 	StringBuilder mtdInst = new StringBuilder();
			mtdInst.append("CP" + Lops.OPERAND_DELIMITOR + "createvar");
				mtdInst.append(Lops.OPERAND_DELIMITOR + outLabels.get(i) + Lops.DATATYPE_PREFIX + matrices.get(i).getDataType() + Lops.VALUETYPE_PREFIX + matrices.get(i).getValueType());
		 		mtdInst.append(Lops.OPERAND_DELIMITOR + outputs[i] + Lops.DATATYPE_PREFIX + DataType.SCALAR + Lops.VALUETYPE_PREFIX + ValueType.STRING);
		 		mtdInst.append(Lops.OPERAND_DELIMITOR + OutputInfo.outputInfoToString(outputInfo[i]) ) ;
			
		  	try {
				b2cinst.add(CPInstructionParser.parseSingleInstruction(mtdInst.toString()));
			} catch (Exception e) {
				throw new PackageRuntimeException(e);
			}
		}
	
		// Finally, generate GMR instruction that performs block2cell conversion
		gmrInst.setGMRInstructions(inputs, inputInfo, numRows, numCols,
				numRowsPerBlock, numColsPerBlock, "", "", "", "", outputs,
				outputInfo, resultIndex, resultDimsUnknown, 0, 1, inLabels,
				outLabels);
			
		b2cinst.add(gmrInst);
	
		// generate instructions that rename the output variables of GMR job
		for (int i = 0; i < matrices.size(); i++) {
			try {
				String s = "CP" + Lops.OPERAND_DELIMITOR + "mvvar"+ Lops.OPERAND_DELIMITOR+ outLabels.get(i) + Lops.OPERAND_DELIMITOR + matrices.get(i).getName(); 
				b2cinst.add(CPInstructionParser.parseSingleInstruction(s));
			} catch (Exception e) {
				throw new PackageRuntimeException(e);
			}
		}
	
		//print instructions
		if (DMLScript.DEBUG) {
			System.out.println("--- Block-2-Cell Instructions ---");
			for(Instruction i : b2cinst) {
				System.out.println(i.toString());
			}
		}
		return b2cinst;
	}

	/**
	 * Method to execute an external function invocation instruction.
	 * 
	 * @param inst
	 * @param dQueue
	 * @throws NimbleCheckedRuntimeException
	 * @throws DMLRuntimeException 
	 */

	public void executeInstruction(ExternalFunctionInvocationInstruction inst,
			DAGQueue dQueue) throws NimbleCheckedRuntimeException, DMLRuntimeException {

		String className = inst.getClassName();
		String configFile = inst.getConfigFile();

		if (className == null)
			throw new PackageRuntimeException("Class name can't be null");

		// create instance of package function.

		Object o;
		try {
			o = Class.forName(className).newInstance();
		} catch (Exception e) {
			throw new PackageRuntimeException(
					"Error generating package function object " + e.toString());
		}

		if (!(o instanceof PackageFunction))
			throw new PackageRuntimeException(
					"Class is not of type PackageFunction");

		PackageFunction func = (PackageFunction) o;

		// add inputs to this package function based on input parameter
		// and their mappings.
		setupInputs(func, inst.getInputParams(), this.getVariables());
		func.setConfiguration(configFile);

		AbstractTask t = null;

		// determine exec location, default is control node
		// and allocate the appropriate NIMBLE task
		if (inst.getExecLocation().compareTo(WORKER) == 0)
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
		if (inst.getExecLocation().compareTo(WORKER) == 0)
			returnFunc = ((WrapperTaskForWorkerNode) t)
			.getUpdatedPackageFunction();
		else
			returnFunc = ((WrapperTaskForControlNode) t)
			.getUpdatedPackageFunction();

		// verify output of function execution matches declaration
		// and add outputs to variableMapping and Metadata
		verifyAndAttachOutputs(returnFunc, inst.getOutputParams());
	}

	/**
	 * Method to verify that function outputs match with declared outputs
	 * 
	 * @param returnFunc
	 * @param outputParams
	 * @throws DMLRuntimeException 
	 */
	protected void verifyAndAttachOutputs(PackageFunction returnFunc,
			String outputParams) throws DMLRuntimeException {

		ArrayList<String> outputs = getParameters(outputParams);
		// make sure they are of equal size first

		if (outputs.size() != returnFunc.getNumFunctionOutputs()) {
			throw new PackageRuntimeException(
					"Function outputs do not match with declaration");
		}

		// iterate over each output and verify that type matches
		for (int i = 0; i < outputs.size(); i++) {
			StringTokenizer tk = new StringTokenizer(outputs.get(i), ":");
			ArrayList<String> tokens = new ArrayList<String>();
			while (tk.hasMoreTokens()) {
				tokens.add(tk.nextToken());
			}

			if (returnFunc.getFunctionOutput(i).getType() == Type.Matrix) {
				Matrix m = (Matrix) returnFunc.getFunctionOutput(i);

				if (!(tokens.get(0)
						.compareTo(getFIODataTypeString(Type.Matrix)) == 0)
						|| !(tokens.get(2).compareTo(
								getMatrixValueTypeString(m.getValueType())) == 0)) {
					throw new PackageRuntimeException(
							"Function output does not match with declaration");
				}

				// add result to variableMapping
				MatrixObjectNew result_matrix = createOutputMatrixObject( m );
				this.getVariables().put(tokens.get(1), result_matrix);
				continue;
			}

			if (returnFunc.getFunctionOutput(i).getType() == Type.Scalar) {
				Scalar s = (Scalar) returnFunc.getFunctionOutput(i);

				if (!tokens.get(0).equals(getFIODataTypeString(Type.Scalar))
						|| !tokens.get(2).equals(
								getScalarValueTypeString(s.getScalarType()))) {
					throw new PackageRuntimeException(
							"Function output does not match with declaration");
				}

				// allocate and set appropriate object based on type
				ScalarObject scalarObject = null;
				ScalarType type = s.getScalarType();
				switch (type) {
				case Integer:
					scalarObject = new IntObject(tokens.get(1),
							Integer.parseInt(s.getValue()));
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
							"Unknown scalar object type");
				}

				this.getVariables().put(tokens.get(1), scalarObject);
				continue;
			}

			if (returnFunc.getFunctionOutput(i).getType() == Type.Object) {
				if (!tokens.get(0).equals(getFIODataTypeString(Type.Object))) {
					new PackageRuntimeException(
							"Function output does not match with declaration");
				}

				throw new PackageRuntimeException(
						"Object types not yet supported");

				// continue;
			}

			throw new PackageRuntimeException(
					"Should never come here -- unknown output type");
		}
	}

	protected MatrixObjectNew createOutputMatrixObject( Matrix m ) 
	{
		MatrixCharacteristics mc = new MatrixCharacteristics(m.getNumRows(),m.getNumCols(), 0, 0);
		//MatrixDimensionsMetaData mtd = new MatrixDimensionsMetaData(mc);
		MatrixFormatMetaData mfmd = new MatrixFormatMetaData(mc, OutputInfo.TextCellOutputInfo, InputInfo.TextCellInputInfo);
		try {
			return new MatrixObjectNew(ValueType.DOUBLE, m.getFilePath(), mfmd);
		} catch (CacheOutOfMemoryException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (CacheStatusException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;		
	}

	/**
	 * Method to get string representation of scalar value type
	 * 
	 * @param scalarType
	 * @return
	 */

	protected String getScalarValueTypeString(ScalarType scalarType) {

		if (scalarType.equals(ScalarType.Double))
			return "Double";
		if (scalarType.equals(ScalarType.Integer))
			return "Integer";
		if (scalarType.equals(ScalarType.Boolean))
			return "Boolean";
		if (scalarType.equals(ScalarType.Text))
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
		ArrayList<FIO> inputObjects = getInputObjects(inputs, variableMapping);
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

	protected ArrayList<FIO> getInputObjects(ArrayList<String> inputs,
			LocalVariableMap variableMapping) {
		ArrayList<FIO> inputObjects = new ArrayList<FIO>();

		for (int i = 0; i < inputs.size(); i++) {
			ArrayList<String> tokens = new ArrayList<String>();
			StringTokenizer tk = new StringTokenizer(inputs.get(i), ":");
			while (tk.hasMoreTokens()) {
				tokens.add(tk.nextToken());
			}

			if (tokens.get(0).equals("Matrix")) {
				String varName = tokens.get(1);
				MatrixObjectNew mobj = (MatrixObjectNew) variableMapping.get(varName);
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
				bObject obj = new bObject(o);
				inputObjects.add(obj);

			}
		}

		return inputObjects;

	}

	protected void modifyInputMatrix(Matrix m, MatrixObjectNew mobj) 
	{
		//do nothing, intended for extensions
	}

	/**
	 * Converts string representation of scalar value type to enum type
	 * 
	 * @param string
	 * @return
	 */
	protected ScalarType getScalarValueType(String string) {
		if (string.equals("Double"))
			return ScalarType.Double;
		if (string.equals("Integer"))
			return ScalarType.Integer;
		if (string.equals("Boolean"))
			return ScalarType.Boolean;
		if (string.equals("String"))
			return ScalarType.Text;

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

	protected com.ibm.bi.dml.packagesupport.Matrix.ValueType getMatrixValueType(String string) {

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
	protected String getFIODataTypeString(Type t) {
		if (t.equals(Type.Matrix))
			return "Matrix";

		if (t.equals(Type.Scalar))
			return "Scalar";

		if (t.equals(Type.Object))
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
		System.out.println("***** INSTRUCTION BLOCK *****");
		for (Instruction i : this._inst) {
			i.printMe();
		}
	}
	
	public HashMap<String,String> getOtherParams()
	{
		return _otherParams;
	}
}