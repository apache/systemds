/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;


public abstract class SQLInstructionBase extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static int INSTRUCTION_ID = 0;
	
	public SQLInstructionBase()
	{
		id = INSTRUCTION_ID++;
	}
	
	int id;
	
	public int getId() {
		return id;
	}

	public abstract ExecutionResult execute(ExecutionContext ec) throws DMLRuntimeException ;
}
