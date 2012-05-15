package com.ibm.bi.dml.test.components.hops;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.utils.HopsException;


public class LiteralOpTest {

	@Test
	public void testConstructLops() {

		LiteralOp lit_hop_d = new LiteralOp("DOUBLE LitOp", 10.0);
		assertEquals(DataType.SCALAR, lit_hop_d.get_dataType());
		assertEquals(ValueType.DOUBLE, lit_hop_d.get_valueType());

		try {
			lit_hop_d.constructLops();
			assertEquals(
					"File_Name: null Label: 10.0 Operation: = READ Format: BINARY Datatype: SCALAR Valuetype: DOUBLE num_rows = -1 num_cols = -1",
					lit_hop_d.get_lops().toString());
		} catch (HopsException e) {
			assertEquals(e.getMessage(),
					"unexpected value type constructing lops.\n");
		}
	}
}
