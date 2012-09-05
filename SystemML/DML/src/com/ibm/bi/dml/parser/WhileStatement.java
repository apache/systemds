package com.ibm.bi.dml.parser;

import java.util.ArrayList;

import com.ibm.bi.dml.utils.LanguageException;


public class WhileStatement extends Statement{
	
	
	private ConditionalPredicate _predicate;
	private ArrayList<StatementBlock> _body;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for WhileStatement");
	}
	
	public WhileStatement(){
		 _predicate = null;
		 _body = new ArrayList<StatementBlock>();
	}
	
	public void setPredicate(ConditionalPredicate pred){
		_predicate = pred;
	}
	
	
	public void addStatementBlock(StatementBlock sb){
		_body.add(sb);
	}
	
	public ConditionalPredicate getConditionalPredicate(){
		return _predicate;
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
		sb.append("while ( ");
		sb.append(_predicate.toString());
		sb.append(") { \n");
		for (StatementBlock block : _body){
			sb.append(block.toString());
		}
		sb.append("}\n");
		return sb.toString();
	}

	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for WhileStatement");
	}
	
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for WhileStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		System.out.println(this.printWarningLocation() + "should not call variablesRead from WhileStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		System.out.println(this.printWarningLocation() + "should not call variablesRead from WhileStatement ");
		return new VariableSet();
	}
}
