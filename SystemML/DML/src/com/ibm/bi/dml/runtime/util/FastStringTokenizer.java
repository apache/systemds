/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.util.NoSuchElementException;

/**
 * This string tokenizer is essentially a simplified copy of the IBM JDK StringTokenizer. 
 * In addition to the default functionality it allows to reset the tokenizer and it makes
 * the simplifying assumptions of (1) no returns delimiter, and (2) single character delimiter.
 * 
 */
public class FastStringTokenizer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
	                                         "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _string = null;
    private char   _del    = 0;
    private int    _pos    = -1;

    /**
     * Constructs a new StringTokenizer for string using the specified
     * delimiters, returnDelimiters is false.
     * 
     * @param string
     *            the string to be tokenized
     * @param delimiters
     *            the delimiters to use
     */
    public FastStringTokenizer(char delimiter) 
    {
        _del = delimiter;
        reset( null );
    }

    /**
     * 
     * @param string
     */
    public void reset( String string )
    {
    	_string = string;
    	_pos = 0;
    }
    
    /**
     * Returns the next token in the string as a String.
     * 
     * @return next token in the string as a String
     * @exception NoSuchElementException
     *                if no tokens remain
     */
    public String nextToken() 
    {
    	int len = _string.length();
    	int start = _pos;	
    	
    	//find start (skip over leading delimiters)
    	while(start < len && _del == _string.charAt(start) )
    		start++;
    	
    	//find end (next delimiter) and return
    	if(start < len) {
        	_pos = _string.indexOf(_del, start);
        	if( start < _pos && _pos < len )
        		return _string.substring(start, _pos);
        	else 
        		return _string.substring(start);
        }
  
    	//no next token
		throw new NoSuchElementException();
    }
    
    ////////////////////////////////////////
    // Custom parsing methods for textcell
    ////////////////////////////////////////
    
    public int nextInt()
    {
    	return Integer.parseInt( nextToken() );
    }
    
    public long nextLong()
    {
    	return Long.parseLong( nextToken() );
    }
    
    public double nextDouble()
    {
    	return Double.parseDouble( nextToken() );
    }
    
    /*
    public void parseMatrixCell( IJV target, String value )
    {
    	reset( value );
    	target.i = nextInt();
    	target.j = nextInt();
    	target.v = nextDouble();
    }
    */
}
