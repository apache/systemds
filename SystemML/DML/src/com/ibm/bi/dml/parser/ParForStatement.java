package com.ibm.bi.dml.parser;

/**
 * This ParForStatement is essentially identical to a ForStatement, except an extended
 * toString method for printing the 'parfor' keyword.
 * 
 *
 */
public class ParForStatement extends ForStatement
{
	@Override
	public String toString() 
	{
		StringBuffer sb = new StringBuffer();
		sb.append("parfor ");
		sb.append(_predicate.toString());
		sb.append(" { \n");
		for (StatementBlock block : _body){
			sb.append(block.toString());
		}
		sb.append("}\n");
		return sb.toString();
	}
} 
 
