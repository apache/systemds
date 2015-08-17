/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.spark.data;

import java.util.ArrayList;
import java.util.List;

import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;

public abstract class LineageObject 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	//basic lineage information
	protected int _numRef = -1;
	protected List<LineageObject> _childs = null;
	protected String _varName = null;
	
	//N:1 back reference to matrix object
	protected MatrixObject _mo = null;
	
	protected LineageObject()
	{
		_numRef = 0;
		_childs = new ArrayList<LineageObject>();
	}
	
	public String getVarName() {
		return _varName;
	}
	
	public int getNumReferences()
	{
		return _numRef;
	}
	
	public void setBackReference(MatrixObject mo)
	{
		_mo = mo;
	}
	
	public boolean hasBackReference()
	{
		return (_mo != null);
	}
	
	public void incrementNumReferences()
	{
		_numRef++;
	}
	
	public void decrementNumReferences()
	{
		_numRef--;
	}
	
	public List<LineageObject> getLineageChilds()
	{
		return _childs;
	}
	
	public void addLineageChild(LineageObject lob)
	{
		lob.incrementNumReferences();
		_childs.add( lob );
	}
}
