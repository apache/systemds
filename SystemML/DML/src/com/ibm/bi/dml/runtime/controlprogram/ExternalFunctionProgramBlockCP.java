package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import org.nimble.exception.NimbleCheckedRuntimeException;

import com.ibm.bi.dml.packagesupport.ExternalFunctionInvocationInstruction;
import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.ExternalFunctionStatement;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.DMLRuntimeException;

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
			Vector<DataIdentifier> inputParams,
			Vector<DataIdentifier> outputParams,
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
				executeInstruction( inst );
			}
			catch (Exception e){
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating instruction " + i + " in external function programBlock. inst: " + inst.toString() );
			}
		}
		
	}
	
	/**
	 * Executes the external function instruction without the use of NIMBLE tasks.
	 * 
	 * @param inst
	 * @throws DMLRuntimeException 
	 * @throws NimbleCheckedRuntimeException
	 */
	@SuppressWarnings("unchecked")
	public void executeInstruction(ExternalFunctionInvocationInstruction inst) throws DMLRuntimeException 
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
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Error generating package function object " + e.toString() );
		}

		if (!(o instanceof PackageFunction))
			throw new PackageRuntimeException(this.printBlockErrorLocation() + "Class is not of type PackageFunction");

		PackageFunction func = (PackageFunction) o;

		// add inputs to this package function based on input parameter
		// and their mappings.
		setupInputs(func, inst.getInputParams(), this.getVariables());
		func.setConfiguration(configFile);
		func.setBaseDir(_baseDir);
		
		//executes function
		func.execute();
		
		// verify output of function execution matches declaration
		// and add outputs to variableMapping and Metadata
		verifyAndAttachOutputs(func, inst.getOutputParams());
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
	protected void modifyInputMatrix(Matrix m, MatrixObjectNew mobj) 
	{
		//pass in-memory object to external function
		m.setMatrixObject( mobj );
	}
	
	@Override
	protected MatrixObjectNew createOutputMatrixObject(Matrix m)
	{
		MatrixObjectNew ret = m.getMatrixObject();
		
		if( ret == null ) //otherwise, pass in-memory matrix from extfunct back to invoking program
		{
			MatrixCharacteristics mc = new MatrixCharacteristics(m.getNumRows(),m.getNumCols(), DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			MatrixFormatMetaData mfmd = new MatrixFormatMetaData(mc, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			ret = new MatrixObjectNew(ValueType.DOUBLE, m.getFilePath(), mfmd);
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

}