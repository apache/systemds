package org.apache.sysds.test.functions.builtin.BuiltinGetSetNamesTest;

import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.junit.Test;

import static org.junit.Assert.*;

public class BuiltinGetSetNamesTest {
    @Test
    public void testGetDefaultNames() {
        FrameBlock fb = new FrameBlock(3, ValueType.STRING);
        FrameBlock names = fb.getNames();
        assertEquals("C1", names.getString(0, 0));
        assertEquals("C2", names.getString(0, 1));
        assertEquals("C3", names.getString(0, 2));
    }

    @Test
    public void testSetAndGetCustomNames() {
        FrameBlock fb = new FrameBlock(2, ValueType.STRING);
        FrameBlock nameRow = new FrameBlock(2, ValueType.STRING);
        nameRow.appendRow(new String[] {"name", "age"});

        fb.setNames(nameRow);

        FrameBlock result = fb.getNames();
        assertEquals("name", result.getString(0, 0));
        assertEquals("age", result.getString(0, 1));
    }

    @Test(expected = DMLRuntimeException.class)
    public void testSetNamesNullFrame() {
        FrameBlock fb = new FrameBlock(2, ValueType.STRING);
        fb.setNames(null);
    }

    @Test(expected = DMLRuntimeException.class)
    public void testSetNamesWrongRowCount() {
        FrameBlock fb = new FrameBlock(2, ValueType.STRING);
        FrameBlock nameRows = new FrameBlock(2, ValueType.STRING);
        nameRows.appendRow(new String[] {"name", "age"});
        nameRows.appendRow(new String[] {"x", "y"});

        fb.setNames(nameRows);
    }

    @Test(expected = DMLRuntimeException.class)
    public void testSetNamesWrongColCount() {
        FrameBlock fb = new FrameBlock(3, ValueType.STRING);
        FrameBlock nameRow = new FrameBlock(2, ValueType.STRING);
        nameRow.appendRow(new String[] {"a", "b"});

        fb.setNames(nameRow);
    }
}

