/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.sql.sqlcontrolprogram;

import com.ibm.bi.dml.runtime.instructions.Instruction;

public interface ISQLCode extends SQLCodeGeneratable {
	public Instruction[] getInstructions();
}
