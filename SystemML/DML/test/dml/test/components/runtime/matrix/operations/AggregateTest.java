package dml.test.components.runtime.matrix.operations;

import static org.junit.Assert.*;

import org.junit.Test;

import dml.runtime.instructions.MRInstructions.AggregateInstruction;
import dml.utils.DMLException;

public class AggregateTest {

    @Test
    public void testParseOperation() {
       /*
        try {
            assertEquals(Aggregate.SupportedOperation.AGG_SUMATION, Aggregate.parseOperation("a+"));
            assertEquals(Aggregate.SupportedOperation.AGG_PRODUCT, Aggregate.parseOperation("a*"));
            assertEquals(Aggregate.SupportedOperation.AGG_MINIMIZATION, Aggregate.parseOperation("amin"));
            assertEquals(Aggregate.SupportedOperation.AGG_MAXIMIZATION, Aggregate.parseOperation("amax"));
        } catch(DMLUnsupportedOperationException e) {
            fail("Operation parsing failed");
        }
        try {
            Aggregate.parseOperation("wrong");
            fail("Wrong operation gets parsed");
        } catch(DMLUnsupportedOperationException e) { }
        */
    }

    @Test
    public void testParseInstruction() {
      /*  try {
            AggregateInstruction instType = (AggregateInstruction) AggregateInstruction.parseInstruction("a+ 0 1");
            //assertEquals(Aggregate.SupportedOperation.AGG_SUMATION, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);

            instType = (AggregateInstruction) AggregateInstruction.parseInstruction("a* 0 1");
            //assertEquals(Aggregate.SupportedOperation.AGG_PRODUCT, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);

            instType = (AggregateInstruction) AggregateInstruction.parseInstruction("amin 0 1");
            //assertEquals(Aggregate.SupportedOperation.AGG_MINIMIZATION, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);

            instType = (AggregateInstruction) AggregateInstruction.parseInstruction("amax 0 1");
            //assertEquals(Aggregate.SupportedOperation.AGG_MAXIMIZATION, instType.operation);
            assertEquals(0, instType.input);
            assertEquals(1, instType.output);
        } catch (DMLException e) {
            fail("Instruction parsing failed");
        }*/
    }

}
