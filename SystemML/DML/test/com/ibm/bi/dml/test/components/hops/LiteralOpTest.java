/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.hops;

import static org.junit.Assert.*;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class LiteralOpTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Test
	public void testConstructLops() {

		LiteralOp lit_hop_d = new LiteralOp("DOUBLE LitOp", 10.0);
		assertEquals(DataType.SCALAR, lit_hop_d.get_dataType());
		assertEquals(ValueType.DOUBLE, lit_hop_d.get_valueType());

		try {
			lit_hop_d.constructLops();
			assertEquals(
					"File_Name: null Label: 10.0 Operation: = READ Format: BINARY Datatype: SCALAR Valuetype: DOUBLE num_rows = 0 num_cols = 0",
					lit_hop_d.get_lops().toString());
		} catch (DMLException e) {
			assertEquals(e.getMessage(),
					"unexpected value type constructing lops.\n");
		}
	}
}
