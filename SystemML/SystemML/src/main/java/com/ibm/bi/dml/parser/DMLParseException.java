/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.parser;

import java.util.ArrayList;

public class DMLParseException extends ParseException
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	/**
	 * The version identifier for this Serializable class.
	 * Increment only if the <i>serialized</i> form of the
	 * class changes.
	 */
	private static final long serialVersionUID = 1L;
	
	private String _filename;
	
	private ArrayList<DMLParseException> _exceptionList;
	
	public ArrayList<DMLParseException> getExceptionList(){
		return _exceptionList;
	}
	
	public DMLParseException(String fname){
		super();
		_filename = fname;
		_exceptionList = new ArrayList<DMLParseException>();
	}
	
	public DMLParseException(String fname, String msg){
		super(msg);
		_filename = fname;
		_exceptionList = new ArrayList<DMLParseException>();
		_exceptionList.add(this);
	}
	
	public DMLParseException(String fname, Exception e){
		super();
		_filename = fname;
		_exceptionList = new ArrayList<DMLParseException>();
		String newMsg = e.getMessage();
		if (e instanceof ParseException && !(e instanceof DMLParseException)){
			ParseException parseEx = (ParseException)e;
			int beginLine = -1, beginColumn = -1;
			String errorToken = null;
			if (parseEx.currentToken != null){
				beginLine    = parseEx.currentToken.beginLine;
				beginColumn  = parseEx.currentToken.beginColumn;
				errorToken   = parseEx.currentToken.image;
				newMsg =  "ERROR: " + _filename + " -- line " + beginLine + ", column " + beginColumn + " -- " + "Parsing error around token \"" + errorToken + "\"";
			} else {
				newMsg =  "ERROR: " + _filename + " -- line " + beginLine + ", column " + beginColumn + " -- " + "Parsing error with unspecified token";
			}
		}
		
		_exceptionList.add(new DMLParseException(_filename, newMsg));
	}
	
	public int size(){
		return _exceptionList.size();
	}
	
	public void add(Exception e){
		if (e instanceof DMLParseException)
			_exceptionList.addAll(((DMLParseException)e).getExceptionList());
		else
			_exceptionList.add(new DMLParseException(this._filename, e));
	}
}