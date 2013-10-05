/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.SQLInstructions;

import java.sql.SQLException;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.sql.sqlcontrolprogram.ExecutionResult;


public class SQLCommitInstruction extends SQLInstructionBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public SQLCommitInstruction()
	{
		instString = "COMMIT;";
	}
	
	@Override
	public ExecutionResult execute(ExecutionContext ec)
			throws DMLRuntimeException {
		
		try {
			if(!ec.getNzConnector().getConnection().getAutoCommit())
			ec.getNzConnector().getConnection().commit();
		} catch (SQLException e) {
			throw new DMLRuntimeException(e);
		}
		
		return new ExecutionResult();
	}

	@Override
	public byte[] getAllIndexes() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public byte[] getInputIndexes() throws DMLRuntimeException {
		// TODO Auto-generated method stub
		return null;
	}
	
}
