/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.Vector;

import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;


public class FunctionProgramBlock extends ProgramBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected ArrayList<ProgramBlock> _childBlocks;
	protected ArrayList<DataIdentifier> _inputParams;
	protected ArrayList<DataIdentifier> _outputParams;
	
	public FunctionProgramBlock( Program prog, Vector <DataIdentifier> inputParams, Vector <DataIdentifier> outputParams) throws DMLRuntimeException
	{
		super(prog);
		_childBlocks = new ArrayList<ProgramBlock>();
		_inputParams = new ArrayList<DataIdentifier>();
		for (DataIdentifier id : inputParams){
			_inputParams.add(new DataIdentifier(id));
			
		}
		_outputParams = new ArrayList<DataIdentifier>();
		for (DataIdentifier id : outputParams){
			_outputParams.add(new DataIdentifier(id));
		}
	}
	
	public ArrayList<DataIdentifier> getInputParams(){
		return _inputParams;
	}
	
	public ArrayList<DataIdentifier> getOutputParams(){
		return _outputParams;
	}
	
	public void addProgramBlock(ProgramBlock childBlock) {
		_childBlocks.add(childBlock);
	}
	
	public void setChildBlocks( ArrayList<ProgramBlock> pbs)
	{
		_childBlocks = pbs;
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
	@Override
	public void execute(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{	
		//dynamically recompile entire function body (according to function inputs)
		/*//TODO in order to make this really useful we need CHAINED UPDATE of STATISTICS along PBs (see parfor)
		try {
			if(    OptimizerUtils.ALLOW_DYN_RECOMPILATION 
				&& DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID )
			{
				System.out.println("FUNCTION RECOMPILE "+_beginLine);
				Recompiler.recompileProgramBlockHierarchy(_childBlocks, _variables, _tid, true);
				System.out.println("END FUNCTION RECOMPILE "+_beginLine);
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Error recompiling function body.", ex);
		}
		*/
		
		// for each program block
		try {						
			for (int i=0 ; i < this._childBlocks.size() ; i++) {
				ec.updateDebugState(i);
				_childBlocks.get(i).execute(ec);
			}
		}
		catch (Exception e){
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating function program block", e);
		}
		
		// check return values
		checkOutputParameters(ec.getVariables());
	}
	
	/**
	 * 
	 * @param vars
	 */
	protected void checkOutputParameters( LocalVariableMap vars )
	{
		for( DataIdentifier diOut : _outputParams )
		{
			String varName = diOut.getName();
			Data dat = vars.get( varName );
			if( dat == null )
				LOG.error("Function output "+ varName +" is missing.");
			else if( dat.getDataType() != diOut.getDataType() )
				LOG.warn("Function output "+ varName +" has wrong data type: "+dat.getDataType()+".");
			else if( dat.getValueType() != diOut.getValueType() )
				LOG.warn("Function output "+ varName +" has wrong value type: "+dat.getValueType()+".");
			   
		}
	}
	
	public void printMe() {
		
		for (ProgramBlock pb : this._childBlocks){
			pb.printMe();
		}
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in function program block generated from function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
}