/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.mr.RandInstruction;


public class RandTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
    @Test
    public void testParseOperation() {
        /*
    	try {
            assertEquals(Rand.SupportedOperation.RAND, Rand.parseOperation("Rand"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            Rand.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
        try {
        	StringBuilder instruction = new StringBuilder("MR" + Instruction.OPERAND_DELIM + "Rand");
        	instruction.append(Instruction.OPERAND_DELIM + 0);
        	instruction.append(Instruction.OPERAND_DELIM + 1);
        	instruction.append(Instruction.OPERAND_DELIM + "10");
        	instruction.append(Instruction.OPERAND_DELIM + "11");
        	instruction.append(Instruction.OPERAND_DELIM + "2");
        	instruction.append(Instruction.OPERAND_DELIM + "2");
        	instruction.append(Instruction.OPERAND_DELIM + "0.0");
        	instruction.append(Instruction.OPERAND_DELIM + "1.0");
        	instruction.append(Instruction.OPERAND_DELIM + "0.5");
        	instruction.append(Instruction.OPERAND_DELIM + "7");
        	instruction.append(Instruction.OPERAND_DELIM + "uniform");
        	instruction.append(Instruction.OPERAND_DELIM + "scratch_space/_t0/");
        	RandInstruction instType = (RandInstruction)RandInstruction.parseInstruction(instruction.toString());
        	
        	
            //assertEquals(Rand.SupportedOperation.RAND, instType.operation);
            assertEquals(10, instType.getRows());
            assertEquals(11, instType.getCols());
            assertEquals(2, instType.getRowsInBlock());
            assertEquals(2, instType.getColsInBlock());
            assertEquals(0, instType.getInput());
            assertEquals(1, instType.output);
            assertEquals(0.0, instType.getMinValue(), 0);
            assertEquals(1.0, instType.getMaxValue(), 0);
            assertEquals(0.5, instType.getSparsity(), 0);
            assertEquals(7, instType.getSeed(), 0);
            assertEquals("uniform", instType.getProbabilityDensityFunction());
            assertEquals("scratch_space/_t0/", instType.getBaseDir());
        } catch (Exception e) {
            fail("Instruction parsing failed");
        }
    }

}
