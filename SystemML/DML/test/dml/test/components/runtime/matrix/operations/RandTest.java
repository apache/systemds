package dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.MRInstructions.RandInstruction;
import dml.utils.DMLException;

public class RandTest {

    @Test
    public void testParseOperation() {
        fail("RandTest.testParseOperation() is not implemented.");
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
        	StringBuilder instruction = new StringBuilder("Rand");
        	instruction.append(Instruction.OPERAND_DELIM + 0);
        	instruction.append(Instruction.OPERAND_DELIM + 1);
        	instruction.append(Instruction.OPERAND_DELIM + "rows=10");
        	instruction.append(Instruction.OPERAND_DELIM + "cols=11");
        	instruction.append(Instruction.OPERAND_DELIM + "min=0.0");
        	instruction.append(Instruction.OPERAND_DELIM + "max=1.0");
        	instruction.append(Instruction.OPERAND_DELIM + "sparsity=0.5");
        	instruction.append(Instruction.OPERAND_DELIM + "pdf=uniform");
        	RandInstruction instType = (RandInstruction)RandInstruction.parseInstruction(instruction.toString());
            //assertEquals(Rand.SupportedOperation.RAND, instType.operation);
            assertEquals(10, instType.rows);
            assertEquals(11, instType.cols);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);
            assertEquals(0.0, instType.minValue, 0);
            assertEquals(1.0, instType.maxValue, 0);
            assertEquals(0.5, instType.sparsity, 0);
            assertEquals("uniform", instType.probabilityDensityFunction);
        } catch (Exception e) {
            fail("Instruction parsing failed");
        }
    }

}
