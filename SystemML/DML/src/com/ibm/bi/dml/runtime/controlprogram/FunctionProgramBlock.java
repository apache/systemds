package com.ibm.bi.dml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.Vector;

import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


public class FunctionProgramBlock extends ProgramBlock {

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
	
	public void execute(ExecutionContext ec) throws DMLRuntimeException, DMLUnsupportedOperationException{
		
		SymbolTable symb = ec.getSymbolTable();
		
		// for each program block
		for (int i=0; i < this._childBlocks.size(); i++){
			
			SymbolTable childSymb = symb.getChildTable(i);
			childSymb.copy_variableMap(symb.get_variableMap());
			ec.setSymbolTable(childSymb);

			ProgramBlock pb = this._childBlocks.get(i);
			
			//pb._variables = new LocalVariableMap();
			//pb.setVariables(_variables);
			
			try {
				pb.execute(ec);
			}
			catch (Exception e){
				throw new DMLRuntimeException(this.printBlockErrorLocation() + "Error evaluating function body", e);
			}
			
			symb.set_variableMap( ec.getSymbolTable().get_variableMap() );
			ec.setSymbolTable(symb);
			//_variables = pb._variables;
		}
	}
	
	@Override
	public SymbolTable createSymbolTable() {
		SymbolTable st = new SymbolTable(true);
		for (int i=0; i < _childBlocks.size(); i++) {
			st.addChildTable(_childBlocks.get(i).createSymbolTable());
		}
		return st;
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