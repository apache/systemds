/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.util;

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
		rowStart = rs;
		rowEnd = re;
		colStart = cs;
		colEnd = ce;
	}
	
	@Override
	public String toString()
	{
		return "["+rowStart+":"+rowEnd+", "+colStart+":"+colEnd+"]";
	}
}
