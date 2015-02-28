/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.nimble.exception.NimbleCheckedRuntimeException;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.ExternalFunctionStatement;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.udf.ExternalFunctionInvocationInstruction;
import com.ibm.bi.dml.udf.Matrix;
import com.ibm.bi.dml.udf.PackageFunction;
import com.ibm.bi.dml.udf.PackageRuntimeException;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
	 * Executes the external function instruction without the use of NIMBLE tasks.
	 * 
	 * @param inst
	 * @throws DMLRuntimeException 
	 * @throws NimbleCheckedRuntimeException
	 */
	@SuppressWarnings("unchecked")
	public void executeInstruction(ExecutionContext ec, ExternalFunctionInvocationInstruction inst) throws DMLRuntimeException 
	{
		String className = inst.getClassName();
		String configFile = inst.getConfigFile();

		if (className == null)
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Class name can't be null");

		// create instance of package function.
		Object o;
		try 
		{
			Class<Instruction> cla = (Class<Instruction>) Class.forName(className);
			o = cla.newInstance();
		} 
		catch (Exception e) 
		{
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Error generating package function object " ,e );
		}

		if (!(o instanceof PackageFunction))
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Class is not of type PackageFunction");

		PackageFunction func = (PackageFunction) o;

		// add inputs to this package function based on input parameter
		// and their mappings.
		setupInputs(func, inst.getInputParams(), ec.getVariables());
		func.setConfiguration(configFile);
		func.setBaseDir(_baseDir);
		
		//executes function
		func.execute();
		
		// verify output of function execution matches declaration
		// and add outputs to variableMapping and Metadata
		verifyAndAttachOutputs(ec, func, inst.getOutputParams());
	}
	

	@Override
	protected void createInstructions() 
	{
		_inst = new ArrayList<Instruction>();

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
			MatrixCharacteristics mc = new MatrixCharacteristics(m.getNumRows(),m.getNumCols(), DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
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