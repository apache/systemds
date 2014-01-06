/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;


/**
 * This RewriteConfig represents an instance configuration of a particular rewrite.
 * 
 */
public abstract class RewriteConfig 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum RewriteConfigType {
		BLOCK_SIZE,
		FORMAT_CHANGE,
		EXEC_TYPE,
		DATA_PARTITIONING,
		VECTORIZATION,
		REPLICATION_FACTOR
	}
	
	protected RewriteConfigType _type  = null; //instance rewrite type
	protected int               _value = -1;   //instance configuration value 
	
	
	public RewriteConfig()
	{
		this( null, -1 );
	}
	
	public RewriteConfig( RewriteConfig that )
	{
		this( that._type, that._value );
	}
	
	public RewriteConfig( RewriteConfigType type, int value )
	{
		_type = type;
		_value = value;
	}
	
	public RewriteConfigType getType()
	{
		return _type;
	}
	
	public void setType( RewriteConfigType type )
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
	
	public boolean isType( RewriteConfigType type )
	{
		return _type == type;
	}
	
	public boolean isValue( int value )
	{
		return _value == value;
	}
	
	/////////
	// abstract methods, forced to be implemented in subclasses 

	/**
	 * Returns the set of values configurations values for this particular rewrite.
	 * 
	 * @return
	 */
	public abstract int[] getDefinedValues();
	
	
	/**
	 * Returns the specific interesting property created by the given rewrite
	 * configuration.
	 * 
	 * @return
	 */
	public abstract InterestingProperty getInterestingProperty();
	
	
	/////////
	// overridden beasic methods 
	
	@Override 
	public boolean equals( Object that )
	{
		//type check for later explicit cast
		if( !(that instanceof RewriteConfig ) )
			return false;
		
		RewriteConfig thatconf = (RewriteConfig)that;
		return (_type == thatconf._type && _value == thatconf._value);
	}
	
	@Override
	public String toString() 
	{
		return "RC["+_type+"="+_value + "]";
	}
	
	
	/*
	TODO


	public abstract void applyToHop(Hop hop);

	public abstract boolean isValidForOperator(Hop operator);


	*/
}
