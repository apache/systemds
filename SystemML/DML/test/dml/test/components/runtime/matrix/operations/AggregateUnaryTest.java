package dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import dml.runtime.instructions.MRInstructions.AggregateUnaryInstruction;
import dml.utils.DMLException;


public class AggregateUnaryTest {

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
