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

package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.parser.DataExpression;

public class CSVFileFormatProperties extends FileFormatProperties 
{
	
	private boolean header;
	private String delim;
	private boolean fill;
	private double fillValue;
	private String naStrings;
	
	private boolean sparse;
	
	public CSVFileFormatProperties() {
		super(FileFormat.CSV);
		
		// get the default values for CSV properties from the language layer
		this.header = DataExpression.DEFAULT_DELIM_HAS_HEADER_ROW;
		this.delim = DataExpression.DEFAULT_DELIM_DELIMITER;
		this.fill = DataExpression.DEFAULT_DELIM_FILL;
		this.fillValue = DataExpression.DEFAULT_DELIM_FILL_VALUE;
		this.sparse = DataExpression.DEFAULT_DELIM_SPARSE;
		this.naStrings = null;
	}
	
	public CSVFileFormatProperties(boolean hasHeader, String delim, boolean fill, double fillValue, String naStrings) {
		super(FileFormat.CSV);
		
		this.header = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.fillValue = fillValue;
		this.naStrings = naStrings;
	}

	public CSVFileFormatProperties(boolean hasHeader, String delim, boolean sparse) {
		super(FileFormat.CSV);
		
		this.header = hasHeader;
		this.delim = delim;
		this.sparse = sparse;
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
	
	public String getNAStrings() { 
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
	
}
