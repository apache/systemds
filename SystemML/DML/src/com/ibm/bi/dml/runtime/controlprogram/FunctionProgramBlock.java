package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.Vector;

import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionContext;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class FunctionProgramBlock extends ProgramBlock 
{
	protected ArrayList<ProgramBlock> _childBlocks;
	protected ArrayList<DataIdentifier> _inputParams;
	protected ArrayList<DataIdentifier> _outputParams;
	
	public FunctionProgramBlock( Program prog, Vector <DataIdentifier> inputParams, Vector <DataIdentifier> outputParams) 
		throws DMLRuntimeException
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
	
	public void setChildBlocks( ArrayList<ProgramBlock> pbs){
		_childBlocks = pbs;
	}
	
	public ArrayList<ProgramBlock> getChildBlocks() {
		return _childBlocks;
	}
	
	public void execute(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{		
		//dynamically recompile entire function body (according to function inputs)
		/*//TODO in order to make this really useful we need CHAINED UPDATE of STATISTICS along PBs
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
		
		// execute all program blocks
		try {
			for( ProgramBlock pb : _childBlocks )
			{	
				// TODO: check with Doug, side effect free?
				pb._variables = new LocalVariableMap(); 
				pb.setVariables(_variables);
				
				pb.execute(ec);
				
				_variables = pb._variables;
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error executing function body", e);
		}
		
		//System.out.println("END FUNCTION EXEC ");
	}
	
	public void printMe() 
	{	
		for (ProgramBlock pb : this._childBlocks)
			pb.printMe();
	}
	
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in function program block generated from function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
	
}