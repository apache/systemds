/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.io;

import org.apache.hadoop.io.IntWritable;

public class TaggedInt extends Tagged<IntWritable>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public TaggedInt()
	{
		tag=-1;
		base=new IntWritable();
	}

	public TaggedInt(IntWritable b, byte t) {
		super(b, t);
	}
	
	public int hashCode()
	{
		return base.hashCode()+tag;
	}
	
	public int compareTo(TaggedInt other)
	{
		if(this.tag!=other.tag)
			return (this.tag-other.tag);
		else if(this.base.get()!=other.base.get())
			return (this.base.get()-other.base.get());
		return 0;
	}

	public boolean equals(TaggedInt other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		return (this.tag==other.tag && this.base.get()==other.base.get());
	}
	
	public boolean equals(Object other)
	{
	//	LOG.info("calling equals for MatrixCellIndexes!");
		if( !(other instanceof TaggedInt))
			return false;
		return equals((TaggedInt)other);
	}
}
