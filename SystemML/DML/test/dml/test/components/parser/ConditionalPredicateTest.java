package dml.test.components.parser;

import static org.junit.Assert.*;

import java.util.HashMap;

import org.junit.Test;

import dml.parser.ConditionalPredicate;
import dml.parser.DataIdentifier;

public class ConditionalPredicateTest {

    @Test
    public void testVariablesRead() {
        DataIdentifier var = new DataIdentifier("var");
        ConditionalPredicate cpToTest = new ConditionalPredicate(var);
        HashMap<String, DataIdentifier> variablesRead = cpToTest.variablesRead().getVariables();
        
        assertEquals(1, variablesRead.size());
        assertTrue("var is missing", variablesRead.containsKey("var"));
        assertEquals(var, variablesRead.get("var"));
    }

}
