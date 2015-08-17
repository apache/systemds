/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.ArrayList;



public class IfStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private ConditionalPredicate _predicate;
	private ArrayList<StatementBlock> _ifBody;
	private ArrayList<StatementBlock> _elseBody;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should not call rewriteStatement for IfStatement");
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for IfStatement");
	}
	
	public IfStatement(){
		 _predicate = null;
		 _ifBody = new ArrayList<StatementBlock>();
		 _elseBody = new ArrayList<StatementBlock>();
	}
	
	public void setConditionalPredicate(ConditionalPredicate pred){
		_predicate = pred;
	}
	
	
	public void addStatementBlockIfBody(StatementBlock sb){
		_ifBody.add(sb);
	}
	
	public void addStatementBlockElseBody(StatementBlock sb){
		_elseBody.add(sb);
	}
	
	public ConditionalPredicate getConditionalPredicate(){
		return _predicate;
	}
	
	public ArrayList<StatementBlock> getIfBody(){
		return _ifBody;
	}
	
	public ArrayList<StatementBlock> getElseBody(){
		return _elseBody;
	}
	
	public void setIfBody(ArrayList<StatementBlock> body){
		_ifBody = body;
	}
	
	public void setElseBody(ArrayList<StatementBlock> body){
		_elseBody = body;
	}
	
	
	@Override
	public boolean controlStatement() {
		return true;
	}
	
	public void initializeforwardLV(VariableSet activeIn) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for IfStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for IfStatement");
		
	}

	public VariableSet initializebackwardLV(VariableSet lo) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should never call initializeforwardLV for IfStatement");
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for IfStatement");
		
	}

	public void mergeStatementBlocksIfBody(){
		_ifBody = StatementBlock.mergeStatementBlocks(_ifBody);
	}
	
	public void mergeStatementBlocksElseBody(){
		if (!_elseBody.isEmpty())
			_elseBody = StatementBlock.mergeStatementBlocks(_elseBody);
	}
	
	public VariableSet variablesReadIfBody() {
		
		return null;
		
	}
	
	public VariableSet variablesReadElseBody() {
		
		LOG.warn("WARNING: line " + this.getBeginLine() + ", column " + this.getBeginColumn() + " --  should not call variablesReadElseBody from IfStatement ");
		return null;
	}
	
	public  VariableSet variablesUpdatedIfBody() {
		
		LOG.warn("WARNING: line " + this.getBeginLine() + ", column " + this.getBeginColumn() + " --  should not call variablesUpdatedIfBody from IfStatement ");
		return null;
	}
	
	public  VariableSet variablesUpdatedElseBody() {
		
		LOG.warn("WARNING: line " + this.getBeginLine() + ", column " + this.getBeginColumn() + " --  should not call variablesUpdatedElseBody from IfStatement ");
		return null;
	}
	
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("if ( ");
		sb.append(_predicate.toString());
		sb.append(") { \n");
		for (StatementBlock block : _ifBody){
			sb.append(block.toString());
		}
		sb.append("}\n");
		if (!_elseBody.isEmpty()){
			sb.append(" else { \n");
			for (StatementBlock block : _elseBody){
				sb.append(block.toString());
			}
			sb.append("}\n");
		}
		return sb.toString();
	}

	
	public VariableSet variablesKill() {
		return new VariableSet();
	}

	@Override
	public VariableSet variablesRead() {
		LOG.warn("WARNING: line " + this.getBeginLine() + ", column " + this.getBeginColumn() + " --  should not call variablesRead from IfStatement ");
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
		LOG.warn("WARNING: line " + this.getBeginLine() + ", column " + this.getBeginColumn() + " --  should not call variablesUpdated from IfStatement ");
		return null;
	}
}
