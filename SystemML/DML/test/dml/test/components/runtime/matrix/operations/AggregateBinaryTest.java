package dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import dml.runtime.instructions.Instruction;
import dml.runtime.instructions.MRInstructions.AggregateBinaryInstruction;
import dml.utils.DMLException;

public class AggregateBinaryTest {

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
