/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.util;

import java.util.NoSuchElementException;

import com.ibm.bi.dml.runtime.io.jdk8.FloatingDecimal;

/**
 * This string tokenizer is essentially a simplified StringTokenizer. 
 * In addition to the default functionality it allows to reset the tokenizer and it makes
 * the simplifying assumptions of (1) no returns delimiter, and (2) single character delimiter.
 * 
 */
public class FastStringTokenizer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
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
    	//return Double.parseDouble( nextToken() );
    
    	//see nextDoubleForParallel, we use the same double parsing
    	//for sequential and parallel parsing because (1) it is faster (~10%)
    	//and (2) for consistency between sequential and parallel readers
    	
    	return FloatingDecimal.parseDouble(nextToken());	
    }
    
    public double nextDoubleForParallel()
    {
    	//JDK 8 floating decimal, which removes a severe scalability bottleneck
    	//(synchronized static cache) in JDK7
    	return FloatingDecimal.parseDouble(nextToken());
    	
    	/*
    	//return Double.parseDouble( nextToken() );
    	
    	//NOTE: Depending on the platform string-2-double conversions were
    	//the main bottleneck in reading text data. Furthermore, we observed
    	//severe contention on multi-threaded parsing on Linux JDK.
    	// ---
    	//This is a known issue and has been fixed in JDK8.
    	//JDK-7032154 : Performance tuning of sun.misc.FloatingDecimal/FormattedFloatingDecimal
    	
    	// Simple workaround without JDK8 code, however, this does NOT guarantee exactly
    	// the same result due to potential for round off errors. 
    	
    	String val = nextToken();
    	double ret = 0;
    
    	if( UtilFunctions.isSimpleDoubleNumber(val) )
    	{ 
    		int ix = val.indexOf('.'); 
    		if( ix > 0 ) //DOUBLE parsing  
        	{
        		String s1 = val.substring(0, ix);
        		String s2 = val.substring(ix+1);
        		long tmp1 = Long.parseLong(s1);
        		long tmp2 = Long.parseLong(s2);
        		ret = (double)tmp2 / Math.pow(10, s2.length()) + tmp1;
        	}
        	else //LONG parsing and cast to double  
        		ret = (double)Long.parseLong(val);
    	}
    	else 
    	{
    		//fall-back to slow default impl if special characters
    		//e.g., ...E-0X, NAN, +-INFINITY, etc
    		ret = Double.parseDouble( val );
    	}
    	
    	return ret;
    	*/
    }
}
