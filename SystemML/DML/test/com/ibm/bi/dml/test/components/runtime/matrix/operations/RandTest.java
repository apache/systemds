package com.ibm.bi.dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;


public class RandTest {

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
        	instruction.append(Instruction.OPERAND_DELIM + "rows=10");
        	instruction.append(Instruction.OPERAND_DELIM + "cols=11");
        	instruction.append(Instruction.OPERAND_DELIM + "rowsInBlock=2");
        	instruction.append(Instruction.OPERAND_DELIM + "colsInBlock=2");
        	instruction.append(Instruction.OPERAND_DELIM + "min=0.0");
        	instruction.append(Instruction.OPERAND_DELIM + "max=1.0");
        	instruction.append(Instruction.OPERAND_DELIM + "sparsity=0.5");
        	instruction.append(Instruction.OPERAND_DELIM + "seed=7");
        	instruction.append(Instruction.OPERAND_DELIM + "pdf=uniform");
        	instruction.append(Instruction.OPERAND_DELIM + "dir=scratch_space/_t0/");
        	RandInstruction instType = (RandInstruction)RandInstruction.parseInstruction(instruction.toString());
        	
        	
            //assertEquals(Rand.SupportedOperation.RAND, instType.operation);
            assertEquals(10, instType.rows);
            assertEquals(11, instType.cols);
            assertEquals(2, instType.rowsInBlock);
            assertEquals(2, instType.colsInBlock);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);
            assertEquals(0.0, instType.minValue, 0);
            assertEquals(1.0, instType.maxValue, 0);
            assertEquals(0.5, instType.sparsity, 0);
            assertEquals(7, instType.seed, 0);
            assertEquals("uniform", instType.probabilityDensityFunction);
            assertEquals("scratch_space/_t0/", instType.baseDir);
        } catch (Exception e) {
            fail("Instruction parsing failed");
        }
    }

}
