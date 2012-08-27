package com.ibm.bi.dml.parser;

import java.util.ArrayList;

import com.ibm.bi.dml.utils.LanguageException;


public class ForStatement extends Statement
{	
	protected IterablePredicate 		_predicate;
	protected ArrayList<StatementBlock> _body;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		throw new LanguageException("should not call rewriteStatement for ForStatement");
	}
	
	public ForStatement(){
		 _predicate = null;
		 _body = new ArrayList<StatementBlock>();
	}
	
	public void setPredicate(IterablePredicate pred){
		_predicate = pred;
	}
	
	
	public void addStatementBlock(StatementBlock sb){
		_body.add(sb);
	}
	
	public IterablePredicate getIterablePredicate(){
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
		StringBuilder sb = new StringBuilder();
		sb.append("for ");
		sb.append(_predicate.toString());
		sb.append(" { \n");
		for (StatementBlock block : _body){
			sb.append(block.toString());
		}
		sb.append("}\n");
		return sb.toString();
	}

	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		throw new LanguageException("should never call initializeforwardLV for ForStatement");
	}
	
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		throw new LanguageException("should never call initializeforwardLV for ForStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		System.out.println("[W] should not call variablesRead from ForStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		System.out.println("[W] should not call variablesRead from ForStatement ");
		return new VariableSet();
	}
} 
 
