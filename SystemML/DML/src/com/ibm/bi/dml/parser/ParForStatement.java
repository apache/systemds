/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

/**
 * This ParForStatement is essentially identical to a ForStatement, except an extended
 * toString method for printing the 'parfor' keyword.
 * 
 *
 */
public class ParForStatement extends ForStatement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public String toString() 
	{
		StringBuilder sb = new StringBuilder();
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
 
