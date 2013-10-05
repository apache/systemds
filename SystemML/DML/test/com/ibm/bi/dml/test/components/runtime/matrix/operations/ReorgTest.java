/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.matrix.operations;


import org.junit.Test;


public class ReorgTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Test
    public void testParseOperation() {
    	/*
        try {
            assertEquals(Reorg.SupportedOperation.REORG_TRANSPOSE, Reorg.parseOperation("r'"));
            assertEquals(Reorg.SupportedOperation.REORG_VECTOR_DIAG, Reorg.parseOperation("rdiag"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            Reorg.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
    /*    try {
            ReorgInstruction instType = (ReorgInstruction)ReorgInstruction.parseInstruction("r' 0 1");
            //assertEquals(Reorg.SupportedOperation.REORG_TRANSPOSE, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);
            
            instType = (ReorgInstruction)ReorgInstruction.parseInstruction("rdiag 0 1");
            //assertEquals(Reorg.SupportedOperation.REORG_VECTOR_DIAG, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);
        } catch (DMLException e) {
            fail("Instruction parsing failed");
        }*/
    }

}
