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

package org.apache.sysds.runtime.io;

import java.io.Serializable;
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.util.UtilFunctions;

public class FileFormatPropertiesCSV extends FileFormatProperties implements Serializable
{
	protected static final Log LOG = LogFactory.getLog(FileFormatPropertiesCSV.class.getName());
	private static final long serialVersionUID = -2870393360885401604L;
	
	private boolean header;
	private String delim;
	private boolean fill;
	private double fillValue;
	private HashSet<String> naStrings;
	
	private boolean sparse;
	
	public FileFormatPropertiesCSV() {
		// get the default values for CSV properties from the language layer
		this.header = DataExpression.DEFAULT_DELIM_HAS_HEADER_ROW;
		this.delim = DataExpression.DEFAULT_DELIM_DELIMITER;
		this.fill = DataExpression.DEFAULT_DELIM_FILL;
		this.fillValue = DataExpression.DEFAULT_DELIM_FILL_VALUE;
		this.sparse = DataExpression.DEFAULT_DELIM_SPARSE;
		this.naStrings = UtilFunctions.defaultNaString;
		if(LOG.isDebugEnabled())
			LOG.debug("FileFormatPropertiesCSV: " + this.toString());
		
	}
	
	public FileFormatPropertiesCSV(boolean hasHeader, String delim, boolean fill, double fillValue, String naStrings) {
		this();
		this.header = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.fillValue = fillValue;

		this.naStrings = new HashSet<>();
		for(String s: naStrings.split(DataExpression.DELIM_NA_STRING_SEP)){
			this.naStrings.add(s);
		}
		if(LOG.isDebugEnabled())
			LOG.debug("FileFormatPropertiesCSV full settings: " + this.toString());
	}

	public FileFormatPropertiesCSV(boolean hasHeader, String delim, boolean sparse) {
		this();
		this.header = hasHeader;
		this.delim = delim;
		this.sparse = sparse;
		this.naStrings = UtilFunctions.defaultNaString;
		if(LOG.isDebugEnabled()){
			LOG.debug("FileFormatPropertiesCSV medium settings: " + this.toString());
		}
	}

	public boolean hasHeader() {
		return header;
	}

	public void setHeader(boolean hasHeader) {
		this.header = hasHeader;
	}

	public String getDelim() {
		return delim;
	}

	public void setNAStrings(HashSet<String> naStrings) {
		this.naStrings = naStrings;
	}

	public HashSet<String> getNAStrings() { 
		return naStrings;
	}

	public void setDelim(String delim) {
		this.delim = delim;
	}

	public boolean isFill() {
		return fill;
	}

	public void setFill(boolean fill) {
		this.fill = fill;
	}

	public double getFillValue() {
		return fillValue;
	}

	public void setFillValue(double fillValue) {
		this.fillValue = fillValue;
	}

	public boolean isSparse() {
		return sparse;
	}

	public void setSparse(boolean sparse) {
		this.sparse = sparse;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("header " +header);
		sb.append(" delim " + delim);
		sb.append(" fill " + fill);
		sb.append(" fillValue " + fillValue);
		sb.append(" naStrings " + naStrings);
		return sb.toString();
	}
}
