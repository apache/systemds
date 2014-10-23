/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.antlr4;

import java.util.HashMap;
import com.ibm.bi.dml.parser.DMLProgram;

public class StatementInfo {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	public com.ibm.bi.dml.parser.Statement stmt = null;
	
	// Valid only for import statements
	public HashMap<String,DMLProgram> namespaces = null;
	
	// Valid only for function statement
	//public String namespace = DMLProgram.DEFAULT_NAMESPACE;
	public String functionName = "";

}
