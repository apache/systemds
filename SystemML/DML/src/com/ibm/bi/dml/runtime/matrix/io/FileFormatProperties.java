/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.io;

public class FileFormatProperties {
	enum FileFormat { PARTITIONED, CSV, NATIVE };
	
	FileFormat fmt;
	
	FileFormatProperties() {
		fmt = FileFormat.NATIVE;
	}
	
	FileFormatProperties(FileFormat fmt) {
		this.fmt = fmt;
	}
	
	public void setFileFormat(FileFormat fmt) {
		this.fmt = fmt;
	}
}
