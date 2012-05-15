package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.Vector;

import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.utils.LanguageException;


public class FunctionStatement extends Statement{
	
	private ArrayList<StatementBlock> _body;
	protected String _name;
	protected Vector <DataIdentifier> _inputParams, _outputParams;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		throw new LanguageException("should not call rewriteStatement for FunctionStatement");
	}
	
	public FunctionStatement(){
		 _body = new ArrayList<StatementBlock>();
		 _name = null;
		 _inputParams = new Vector<DataIdentifier>();
		 _outputParams = new Vector<DataIdentifier>();
	}
	
	public Vector <DataIdentifier> getInputParams(){
		return _inputParams;
	}
	
	public Vector <DataIdentifier> getOutputParams(){
		return _outputParams;
	}
	
	public void setName(String fname){
		_name = fname;
	}
	
	public String getName(){
		return _name;
	}
	
	public void addStatementBlock(StatementBlock sb){
		_body.add(sb);
	}
	
	public ArrayList<StatementBlock> getBody(){
		return _body;
	}
	
	public void getBody(ArrayList<StatementBlock> body){
		_body = body;
	}
	
	public void setBody(ArrayList<StatementBlock> body){
		_body = body;
	}
	
	
	@Override
	public boolean controlStatement() {
		return true;
	}

	public void mergeStatementBlocks(){
		_body = StatementBlock.mergeStatementBlocks(_body);
	}
	
	public String toString(){
		StringBuffer sb = new StringBuffer();
		sb.append(_name + " = ");
		
		sb.append("function ( ");
		
		for (int i=0; i<_inputParams.size(); i++){
			DataIdentifier curr = _inputParams.get(i);
			sb.append(curr.getName());
			if (curr.getDefaultValue() != null) sb.append(" = " + curr.getDefaultValue());
			if (i < _inputParams.size()-1) sb.append(", ");
		}
		sb.append(") return (");
		
		for (int i=0; i<_outputParams.size(); i++){
			sb.append(_outputParams.get(i).getName());
			if (i < _outputParams.size()-1) sb.append(", ");
		}
		sb.append(") { \n");
		
		for (StatementBlock block : _body){
			sb.append(block.toString());
		}
		sb.append("} \n");
		return sb.toString();
	}

	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		throw new LanguageException("should never call initializeforwardLV for FunctionStatement");
	}
	
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		throw new LanguageException("should never call initializeforwardLV for FunctionStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		System.out.println("[W] should not call variablesRead from FunctionStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		System.out.println("[W] should not call variablesRead from FunctionStatement ");
		return new VariableSet();
	}
	
	public static String[] createFunctionCallVariables( ArrayList<Lops> lops )
	{
		String[] ret = new String[lops.size()]; //vars in order
		
		for( int i=0; i<lops.size(); i++ )
		{	
			Lops llops = lops.get(i);
			if( llops.getType()==Lops.Type.Data )
				ret[i] = llops.getOutputParameters().getLabel(); 
		}
		
		return ret;
	}
}
