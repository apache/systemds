/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.matrix.operations;


import org.junit.Test;



public class AggregateUnaryTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Test
    public void testParseOperation() {
        /*
        try {
            assertEquals(AggregateUnary.SupportedOperation.AGU_SUM, AggregateUnary.parseOperation("ua+"));
            assertEquals(AggregateUnary.SupportedOperation.AGU_ROW_SUM, AggregateUnary.parseOperation("uar+"));
            assertEquals(AggregateUnary.SupportedOperation.AGU_COLUMN_SUM, AggregateUnary.parseOperation("uac+"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            AggregateUnary.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
  /*      try {
            AggregateUnaryInstruction instType = (AggregateUnaryInstruction)AggregateUnaryInstruction.parseInstruction("ua+ 0 1");
            //assertEquals(AggregateUnary.SupportedOperation.AGU_SUM, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);

            instType = (AggregateUnaryInstruction)AggregateUnaryInstruction.parseInstruction("uar+ 0 1");
            //assertEquals(AggregateUnary.SupportedOperation.AGU_ROW_SUM, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);

            instType = (AggregateUnaryInstruction)AggregateUnaryInstruction.parseInstruction("uac+ 0 1");
            //assertEquals(AggregateUnary.SupportedOperation.AGU_COLUMN_SUM, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);
        } catch (DMLException e) {
            fail("Instruction parsing failed");
        }*/
    }

}
