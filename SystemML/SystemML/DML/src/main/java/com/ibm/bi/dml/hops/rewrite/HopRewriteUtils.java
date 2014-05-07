/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.rewrite;

import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.BooleanObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.DoubleObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.StringObject;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class HopRewriteUtils 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
	public static boolean getBooleanValue( LiteralOp op )
		throws HopsException
	{
		switch( op.get_valueType() )
		{
			case DOUBLE:  return op.getDoubleValue() != 0; 
			case INT:	  return op.getLongValue()   != 0;
			case BOOLEAN: return op.getBooleanValue();
			
			default: throw new HopsException("Invalid boolean value: "+op.get_valueType());
		}
	}

	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
	public static double getDoubleValue( LiteralOp op )
		throws HopsException
	{
		switch( op.get_valueType() )
		{
			case DOUBLE:  return op.getDoubleValue(); 
			case INT:	  return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			
			default: throw new HopsException("Invalid double value: "+op.get_valueType());
		}
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
	public static long getIntValue( LiteralOp op )
		throws HopsException
	{
		switch( op.get_valueType() )
		{
			case DOUBLE:  return UtilFunctions.toLong(op.getDoubleValue()); 
			case INT:	  return op.getLongValue();
			case BOOLEAN: return op.getBooleanValue() ? 1 : 0;
			
			default: throw new HopsException("Invalid int value: "+op.get_valueType());
		}
	}
	
	/**
	 * 
	 * @param op
	 * @return
	 * @throws HopsException
	 */
	public static ScalarObject getScalarObject( LiteralOp op )
	{
		ScalarObject ret = null;
		
		try
		{
			switch( op.get_valueType() )
			{
				case DOUBLE:  ret = new DoubleObject(op.getDoubleValue()); break;
				case INT:	  ret = new IntObject((int)op.getLongValue()); break;
				case BOOLEAN: ret = new BooleanObject(op.getBooleanValue()); break;
				case STRING:  ret = new StringObject(op.getStringValue()); break;
			}
		}
		catch(Exception ex)
		{
			throw new RuntimeException("Failed to create scalar object for constant. Continue.", ex);
		}
		
		return ret;
	}
}
