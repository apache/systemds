/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

/**
 * An instance of this class represents one specific data flow property of an operator
 * output. Thus, an operator output is characterized via a set of these properties. 
 * 
 */
public class InterestingProperty 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum InterestingPropertyType {
		BLOCK_SIZE,        //any positive integer value
		FORMAT,            //TEXT, BINARY_CELL, BINARY_BLOCK, MM, CSV
		DATA_LOCATION,     //MEM, HDFS
		PARTITION_FORMAT,  //NONE, ROW, COLUMN, ROW_BLOCK, COLUMN_BLOCK
		REPLICATION        //any positive integer
	}
	
	public enum DataLocationType{
		MEM,
		HDFS,
	}
	
	public enum FormatType{
		BINARY_BLOCK,
		BINARY_CELL,
		TEXT_CELL,
		TEXT_MM,
		TEXT_CSV
	}
	
	//instance data of one particular interesting property
	protected InterestingPropertyType _type  = null;
	protected int                     _value = -1;
	
	public InterestingProperty( )
	{
		this( null, -1 );
	}
	
	public InterestingProperty( InterestingProperty that )
	{
		this( that._type, that._value);
	}
	
	public InterestingProperty( InterestingPropertyType type, int value )
	{
		_type = type;
		_value = value;
	}
	
	public InterestingPropertyType getType()
	{
		return _type;
	}
	
	public void setType( InterestingPropertyType type )
	{
		_type = type;
	}
	
	public int getValue()
	{
		return _value;
	}
	
	public void setValue( int value )
	{
		_value = value;
	}
	
	@Override 
	public boolean equals( Object that )
	{
		//type check for later explicit cast
		if( !(that instanceof InterestingProperty ) )
			return false;
		
		InterestingProperty thatip = (InterestingProperty)that;
		return (_type == thatip._type && _value == thatip._value);
	}
	
	@Override
	public String toString()
	{
		return "IP["+_type+"="+_value + "]";
	}
}
