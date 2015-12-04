/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.api.monitoring;

public class Location implements Comparable<Location> {
	public int beginLine;
	public int endLine;
	public int beginCol;
	public int endCol;
	public Location(int beginLine, int endLine, int beginCol, int endCol) {
		this.beginLine = beginLine;
		this.endLine = endLine;
		this.beginCol = beginCol;
		this.endCol = endCol;
	}
	
	@Override
	public boolean equals(Object other) {
		if(other instanceof Location) {
			Location loc = (Location) other;
			if(loc.beginLine == beginLine && loc.endLine == endLine && loc.beginCol == beginCol && loc.endCol == endCol) {
				return true;
			}
			else
				return false;
		}
		return false;
	}
	
	private int compare(int v1, int v2) {
		return new Integer(v1).compareTo(new Integer(v2));
	}
	
	public String toString() {
		return beginLine + ":" + beginCol + ", " + endLine + ":" + endCol;
	}

	@Override
	public int compareTo(Location loc) {
		if(loc.beginLine == beginLine && loc.endLine == endLine && loc.beginCol == beginCol && loc.endCol == endCol)
			return 0;
		
		int retVal = compare(beginLine, loc.beginLine);
		if(retVal != 0) { 
			return retVal;
		}
		else { 
			retVal = compare(beginCol, loc.beginCol);
			if(retVal != 0) { 
				return retVal;
			}
			else { 
				retVal = compare(endLine, loc.endLine);
				if(retVal != 0) { 
					return retVal;
				}
				else {
					return compare(endCol, loc.endCol);
				}
			}
		}
	}
}
