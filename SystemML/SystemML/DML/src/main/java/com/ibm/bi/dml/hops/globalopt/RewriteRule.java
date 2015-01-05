/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.HashSet;
import java.util.Set;


/**
 * For now just a marker class.
 */
public abstract class RewriteRule implements HopsVisitor
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	protected Set<HopsMetaData> changes = new HashSet<HopsMetaData>();
	public abstract HopsDag rewrite(HopsDag toModify);
	public abstract void applyChanges();
	
	protected void addChange(HopsMetaData meta){
		this.changes.add(meta);
	}
	
	
}
