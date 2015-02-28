/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.sql;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.context.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;


public abstract class SQLInstructionBase extends Instruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private static IDSequence seq = new IDSequence();
	protected int id;
	
	
	public SQLInstructionBase()
	{
		id = (int)seq.getNextID();
	}
	
	public int getId() {
		return id;
	}

	public abstract ExecutionResult execute(ExecutionContext ec) 
		throws DMLRuntimeException;
}
