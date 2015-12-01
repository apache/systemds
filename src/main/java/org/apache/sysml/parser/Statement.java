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

package org.apache.sysml.parser;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public abstract class Statement 
{

	
	protected static final Log LOG = LogFactory.getLog(Statement.class.getName());
	
	public static final String OUTPUTSTATEMENT = "write";
					
	// parameter names for seq()
	public static final String SEQ_FROM = "from"; 
	public static final String SEQ_TO   = "to";
	public static final String SEQ_INCR	= "incr";
		
	public static final String SOURCE  	= "source";
	public static final String SETWD 	= "setwd";

	public static final String MATRIX_DATA_TYPE = "matrix";
	public static final String FRAME_DATA_TYPE = "frame";
	public static final String SCALAR_DATA_TYPE = "scalar";
	
	public static final String DOUBLE_VALUE_TYPE = "double";
	public static final String BOOLEAN_VALUE_TYPE = "boolean";
	public static final String INT_VALUE_TYPE = "int";
	public static final String STRING_VALUE_TYPE = "string";
	
	// String constants related to Grouped Aggregate parameters
	public static final String GAGG_TARGET  = "target";
	public static final String GAGG_GROUPS  = "groups";
	public static final String GAGG_WEIGHTS = "weights";
	public static final String GAGG_FN      = "fn";
	public static final String GAGG_FN_SUM      = "sum";
	public static final String GAGG_FN_COUNT    = "count";
	public static final String GAGG_FN_MEAN     = "mean";
	public static final String GAGG_FN_VARIANCE = "variance";
	public static final String GAGG_FN_CM       = "centralmoment";
	public static final String GAGG_FN_CM_ORDER = "order";
	public static final String GAGG_NUM_GROUPS  = "ngroups";
	
	public abstract boolean controlStatement();
	
	public abstract VariableSet variablesRead();
	public abstract VariableSet variablesUpdated();
 
	public abstract void initializeforwardLV(VariableSet activeIn) throws LanguageException;
	public abstract VariableSet initializebackwardLV(VariableSet lo) throws LanguageException;
	
	public abstract Statement rewriteStatement(String prefix) throws LanguageException;
	
	// Used only insider python parser to allow for ignoring newline logic
	private boolean isEmptyNewLineStatement = false;
	public boolean isEmptyNewLineStatement() {
		return isEmptyNewLineStatement;
	}	
	public void setEmptyNewLineStatement(boolean isEmptyNewLineStatement) {
		this.isEmptyNewLineStatement = isEmptyNewLineStatement;
	}
	
	///////////////////////////////////////////////////////////////////////////
	// store exception info + position information for statements
	///////////////////////////////////////////////////////////////////////////
	
	
	private String _filename;
	private int _beginLine, _beginColumn;
	private int _endLine,   _endColumn;
	
	public void setFilename(String passed)  { _filename = passed;	}
	public void setBeginLine(int passed)    { _beginLine = passed;	}
	public void setBeginColumn(int passed) 	{ _beginColumn = passed;}
	public void setEndLine(int passed) 		{ _endLine = passed;   }
	public void setEndColumn(int passed)	{ _endColumn = passed; }
	
	public void setAllPositions(String filename, int blp, int bcp, int elp, int ecp){
		_filename    = filename;
		_beginLine	 = blp; 
		_beginColumn = bcp; 
		_endLine 	 = elp;
		_endColumn 	 = ecp;
	}

	public int getBeginLine()	{ return _beginLine;   }
	public int getBeginColumn() { return _beginColumn; }
	public int getEndLine() 	{ return _endLine;   }
	public int getEndColumn()	{ return _endColumn; }
	public String getFilename() { return _filename;  }
		
	public String printErrorLocation(){
		return "ERROR: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printWarningLocation(){
		return "WARNING: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printInfoLocation(){
		return "INFO: " + _filename + " -- line " + _beginLine + ", column " + _beginColumn + " -- ";
	}
	
	public String printErrorLocation(int beginLine, int beginColumn){
		return "ERROR: " + _filename + " -- line " + beginLine + ", column " + beginColumn + " -- ";
	}
	
	public String printWarningLocation(int beginLine, int beginColumn){
		return "WARNING: " + _filename + " -- line " + beginLine + ", column " + beginColumn + " -- ";
	}
	
	public String printInfoLocation(int beginLine, int beginColumn){
		return "INFO: " + _filename + " -- line " + beginLine + ", column " + beginColumn + " -- ";
	}

	
}
