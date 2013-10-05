/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.matrix.operations;

import org.junit.Test;

public class AggregateBinaryTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Test
    public void testParseOperation() {
    	/*
    	try {
            assertEquals(AggregateBinary.SupportedOperation.AGB_MMULT, AggregateBinary.parseOperation("ba+*"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            AggregateBinary.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
 /*       try {
        	StringBuilder instruction = new StringBuilder("ba+*");
        	instruction.append(Instruction.OPERAND_DELIM + 0);
        	instruction.append(Instruction.OPERAND_DELIM + 1);
        	instruction.append(Instruction.OPERAND_DELIM + 2);
            AggregateBinaryInstruction instType = (AggregateBinaryInstruction) AggregateBinaryInstruction.parseInstruction(instruction.toString());
            //assertEquals(AggregateBinary.SupportedOperation.AGB_MMULT, instType.operation);
            assertEquals(0, instType.input1);
            assertEquals(1, instType.input2);
            assertEquals(2, instType.output);
        } catch (DMLException e) {
            fail("Instruction parsing failed");
        }*/
    }

}
