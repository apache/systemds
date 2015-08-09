package com.ibm.bi.dml.runtime.util;

import java.io.Serializable;

//start and end are all inclusive
public class IndexRange implements Serializable
{
	private static final long serialVersionUID = 5746526303666494601L;
	public long rowStart=0;
	public long rowEnd=0;
	public long colStart=0;
	public long colEnd=0;
	
	public IndexRange(long rs, long re, long cs, long ce)
	{
		set(rs, re, cs, ce);
	}
	public void set(long rs, long re, long cs, long ce)
	{
		//if(re<rs || ce<cs)
		//	throw new RuntimeException("This is invalid index range: ["+rs+":"+re+", "+cs+":"+ce+"]!");
		rowStart=rs;
		rowEnd=re;
		colStart=cs;
		colEnd=ce;
	}
	
	@Override
	public String toString()
	{
		return "["+rowStart+":"+rowEnd+", "+colStart+":"+colEnd+"]";
	}
}
