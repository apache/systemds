/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;



public class ForStatement extends Statement
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected IterablePredicate 		_predicate;
	protected ArrayList<StatementBlock> _body;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should not call rewriteStatement for ForStatement");
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for ForStatement");
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
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for ForStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for ForStatement");
	}
	
	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for ForStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for ForStatement");
		
	}
	
	@Override
	public VariableSet variablesRead() {
		LOG.error(this.printErrorLocation() + "should not call variablesRead from ForStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		LOG.error(this.printErrorLocation() +  "should not call variablesRead from ForStatement ");
		return new VariableSet();
	}
} 
 
