/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;
import java.util.Vector;

import com.ibm.bi.dml.lops.Lop;


public class FunctionStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private ArrayList<StatementBlock> _body;
	protected String _name;
	protected Vector <DataIdentifier> _inputParams, _outputParams;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should not call rewriteStatement for FunctionStatement");
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for FunctionStatement");
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
	
	public void setInputParams(Vector <DataIdentifier> inputParams){
		_inputParams = inputParams;
	}
	
	public void setOutputParams(Vector <DataIdentifier> outputParams){
		_outputParams = outputParams;
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
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for FunctionStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for FunctionStatement");
	}
	
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for FunctionStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for FunctionStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		LOG.warn(this.printWarningLocation() + " -- should not call variablesRead from FunctionStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		LOG.warn(this.printWarningLocation() + " -- should not call variablesRead from FunctionStatement ");
		return new VariableSet();
	}
	
	public static String[] createFunctionCallVariables( ArrayList<Lop> lops )
	{
		String[] ret = new String[lops.size()]; //vars in order
		
		for( int i=0; i<lops.size(); i++ )
		{	
			Lop llops = lops.get(i);
			if( llops.getType()==Lop.Type.Data )
				ret[i] = llops.getOutputParameters().getLabel(); 
		}
		
		return ret;
	}
}
