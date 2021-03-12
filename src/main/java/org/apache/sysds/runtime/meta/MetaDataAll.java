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

package org.apache.sysds.runtime.meta;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.parser.BooleanIdentifier;
import org.apache.sysds.parser.ConstIdentifier;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.DoubleIdentifier;
import org.apache.sysds.parser.Expression;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.parser.StringIdentifier;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.utils.JSONHelper;
import org.apache.wink.json4j.JSONArray;
import org.apache.wink.json4j.JSONException;
import org.apache.wink.json4j.JSONObject;

public class MetaDataAll extends DataIdentifier {
//	private static final Log LOG = LogFactory.getLog(DataExpression.class.getName());
//
//	public static final String RAND_DIMS = "dims";
//
//	public static final String RAND_ROWS = "rows";
//	public static final String RAND_COLS = "cols";
//	public static final String RAND_MIN = "min";
//	public static final String RAND_MAX = "max";
//	public static final String RAND_SPARSITY = "sparsity";
//	public static final String RAND_SEED = "seed";
//	public static final String RAND_PDF = "pdf";
//	public static final String RAND_LAMBDA = "lambda";
//
//	public static final String RAND_PDF_UNIFORM = "uniform";
//
//	public static final String RAND_BY_ROW = "byrow";
//	public static final String RAND_DIMNAMES = "dimnames";
//	public static final String RAND_DATA = "data";
//
//	public static final String IO_FILENAME = "iofilename";
//	public static final String READROWPARAM = "rows";
//	public static final String READCOLPARAM = "cols";
//	public static final String READNNZPARAM = "nnz";
//
//	public static final String SQL_CONN = "conn";
//	public static final String SQL_USER = "user";
//	public static final String SQL_PASS = "password";
//	public static final String SQL_QUERY = "query";
//
//	public static final String FED_ADDRESSES = "addresses";
//	public static final String FED_RANGES = "ranges";
//	public static final String FED_TYPE = "type";
//
//	public static final String FORMAT_TYPE = "format";
//
//	public static final String ROWBLOCKCOUNTPARAM = "rows_in_block";
//	public static final String COLUMNBLOCKCOUNTPARAM = "cols_in_block";
//	public static final String DATATYPEPARAM = "data_type";
//	public static final String VALUETYPEPARAM = "value_type";
//	public static final String DESCRIPTIONPARAM = "description";
//	public static final String AUTHORPARAM = "author";
//	public static final String SCHEMAPARAM = "schema";
//	public static final String CREATEDPARAM = "created";
//
//	public static final String PRIVACY = "privacy";
//	public static final String FINE_GRAINED_PRIVACY = "fine_grained_privacy";
//
//	// Parameter names relevant to reading/writing delimited/csv files
//	public static final String DELIM_DELIMITER = "sep";
//	public static final String DELIM_HAS_HEADER_ROW = "header";
//	public static final String DELIM_FILL = "fill";
//	public static final String DELIM_FILL_VALUE = "default";
//	public static final String DELIM_NA_STRINGS = "naStrings";
//	public static final String DELIM_NA_STRING_SEP = "\u00b7";
//
//
//	public static final String DELIM_SPARSE = "sparse";  // applicable only for write
//
//	/** Valid parameter names in metadata file */
//	public static final Set<String> READ_VALID_MTD_PARAM_NAMES =new HashSet<>(
//		Arrays.asList(IO_FILENAME, READROWPARAM, READCOLPARAM, READNNZPARAM,
//			FORMAT_TYPE, ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, DATATYPEPARAM,
//			VALUETYPEPARAM, SCHEMAPARAM, DESCRIPTIONPARAM, AUTHORPARAM, CREATEDPARAM,
//			// Parameters related to delimited/csv files.
//			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW, DELIM_NA_STRINGS,
//			// Parameters related to privacy
//			PRIVACY, FINE_GRAINED_PRIVACY));
//
//	/** Valid parameter names in arguments to read instruction */
//	public static final Set<String> READ_VALID_PARAM_NAMES = new HashSet<>(
//		Arrays.asList(IO_FILENAME, READROWPARAM, READCOLPARAM, FORMAT_TYPE, DATATYPEPARAM,
//			VALUETYPEPARAM, SCHEMAPARAM, ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, READNNZPARAM,
//			// Parameters related to delimited/csv files.
//			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW, DELIM_NA_STRINGS));
//
//	/* Default Values for delimited (CSV/LIBSVM) files */
//	public static final String  DEFAULT_DELIM_DELIMITER = ",";
//	public static final boolean DEFAULT_DELIM_HAS_HEADER_ROW = false;
//	public static final boolean DEFAULT_DELIM_FILL = true;
//	public static final double  DEFAULT_DELIM_FILL_VALUE = 0.0;
//	public static final boolean DEFAULT_DELIM_SPARSE = false;
//	public static final String  DEFAULT_NA_STRINGS = "";
//	public static final String  DEFAULT_SCHEMAPARAM = "NULL";
//
//	private Expression.DataOp _opcode;
//	private HashMap<String, Expression> _varParams;
//	private boolean _strInit = false; //string initialize
//	private boolean _checkMetadata = true; // local skip meta data reads

	// TODO added
	private JSONObject _metaObj;

	public String _formatTypeString;

//
//	//csv
//	private String _delimiter = null;
//
//	public int _rowsInBlock;
//	public int _colsInBlock;
//
//	public Types.DataType _dataType;
//	public Types.ValueType _valueType;
//	public String _schema;
//
//	public boolean _header;
//	public boolean _sparse;
//	public boolean _fill;
//
//	public String _privacy;

	public MetaDataAll() {
		// do nothing
	}

	public MetaDataAll(String mtdFileName, boolean conditional) {
		_metaObj = readMetadataFile(mtdFileName, conditional);
		setFormatTypeString(null);

		//TODO parse the most important params
	}

	public void setFormatTypeString(String format) {
		_formatTypeString = _formatTypeString != null && format == null && _metaObj != null ? (String)JSONHelper.get(_metaObj, DataExpression.FORMAT_TYPE) : format ;
		if( Types.FileFormat.isDelimitedFormat(this._formatTypeString) )
			this.setFileFormat(Types.FileFormat.safeValueOf(_formatTypeString));
	}

	public JSONObject readMetadataFile(String filename, boolean conditional)
	{
		JSONObject retVal = null;
		boolean exists = HDFSTool.existsFileOnHDFS(filename);
		boolean isDir = HDFSTool.isDirectory(filename);

		// CASE: filename is a directory -- process as a directory
		if( exists && isDir )
		{
			retVal = new JSONObject();
			for(FileStatus stat : HDFSTool.getDirectoryListing(filename)) {
				Path childPath = stat.getPath(); // gives directory name
				if( !childPath.getName().startsWith("part") )
					continue;
				try (BufferedReader br = new BufferedReader(new InputStreamReader(
					IOUtilFunctions.getFileSystem(childPath).open(childPath))))
				{
					JSONObject childObj = JSONHelper.parse(br);
					for( Object obj : childObj.entrySet() ){
						@SuppressWarnings("unchecked") Map.Entry<Object,Object> e = (Map.Entry<Object, Object>) obj;
						Object key = e.getKey();
						Object val = e.getValue();
						retVal.put(key, val);
					}
				}
				catch(Exception e){
					raiseValidateError("for MTD file in directory, error parting part of MTD file with path " + childPath.toString() + ": " + e.getMessage(), conditional);
				}
			}
		}

		// CASE: filename points to a file
		else if (exists) {
			Path path = new Path(filename);
			try (BufferedReader br = new BufferedReader(new InputStreamReader(
				IOUtilFunctions.getFileSystem(path).open(path))))
			{
				retVal = new JSONObject(br);
			}
			catch (Exception e){
				raiseValidateError("error parsing MTD file with path " + filename + ": " + e.getMessage(), conditional);
			}
		}

		return retVal;
	}

	public boolean mtdExists() { return _metaObj != null; }

	public JSONObject getMetaObject() { return _metaObj; }


	@SuppressWarnings("unchecked")
	public HashMap<String, Expression> parseMetaDataFileParameters(String mtdFileName, boolean conditional, HashMap<String, Expression> varParams)
	{
		for( Object obj : _metaObj.entrySet() ){
			Map.Entry<Object,Object> e = (Map.Entry<Object, Object>) obj;
			Object key = e.getKey();
			Object val = e.getValue();

			boolean isValidName = DataExpression.READ_VALID_MTD_PARAM_NAMES.contains(key);

			if (!isValidName){ //wrong parameters always rejected
				raiseValidateError("MTD file " + mtdFileName + " contains invalid parameter name: " + key, false);
			}

			// if the read method parameter is a constant, then verify value matches MTD metadata file
			if (varParams.get(key.toString()) != null && (varParams.get(key.toString()) instanceof ConstIdentifier)
				&& !varParams.get(key.toString()).toString().equalsIgnoreCase(val.toString())) {
				raiseValidateError("Parameter '" + key.toString()
					+ "' has conflicting values in metadata and read statement. MTD file value: '"
					+ val.toString() + "'. Read statement value: '" + varParams.get(key.toString()) + "'.", conditional);
			} else {
				// if the read method does not specify parameter value, then add MTD metadata file value to parameter list
				if (varParams.get(key.toString()) == null){
					if (( !key.toString().equalsIgnoreCase(DataExpression.DESCRIPTIONPARAM) ) &&
						( !key.toString().equalsIgnoreCase(DataExpression.AUTHORPARAM) ) &&
						( !key.toString().equalsIgnoreCase(DataExpression.CREATEDPARAM) ) )
					{
						StringIdentifier strId = new StringIdentifier(val.toString(), this);

						if ( key.toString().equalsIgnoreCase(DataExpression.DELIM_HAS_HEADER_ROW)
							|| key.toString().equalsIgnoreCase(DataExpression.DELIM_FILL)
							|| key.toString().equalsIgnoreCase(DataExpression.DELIM_SPARSE)
						) {
							// parse these parameters as boolean values
							BooleanIdentifier boolId = null;
							if (strId.toString().equalsIgnoreCase("true")) {
								boolId = new BooleanIdentifier(true, this);
							} else if (strId.toString().equalsIgnoreCase("false")) {
								boolId = new BooleanIdentifier(false, this);
							} else {
								raiseValidateError("Invalid value provided for '" + DataExpression.DELIM_HAS_HEADER_ROW + "' in metadata file '" + mtdFileName + "'. "
									+ "Must be either TRUE or FALSE.", conditional);
							}
							varParams.remove(key.toString());
							addVarParam(key.toString(), boolId, varParams);

							switch(key.toString().toUpperCase()) {
								case DataExpression.DELIM_HAS_HEADER_ROW: ;
								case DataExpression.DELIM_FILL: ;
								case DataExpression.DELIM_SPARSE: ;
							}

						}
						else if ( key.toString().equalsIgnoreCase(DataExpression.DELIM_FILL_VALUE)) {
							// parse these parameters as numeric values
							DoubleIdentifier doubleId = new DoubleIdentifier(Double.parseDouble(strId.toString()),
								this);
							varParams.remove(key.toString());
							addVarParam(key.toString(), doubleId, varParams);
						}
						else if (key.toString().equalsIgnoreCase(DataExpression.DELIM_NA_STRINGS)
							|| key.toString().equalsIgnoreCase(DataExpression.PRIVACY)
							|| key.toString().equalsIgnoreCase(DataExpression.FINE_GRAINED_PRIVACY)) {
							String naStrings = null;
							if ( val instanceof String) {
								naStrings = val.toString();
							}
							else if (val instanceof JSONArray) {
								StringBuilder sb = new StringBuilder();
								JSONArray valarr = (JSONArray)val;
								for(int naid=0; naid < valarr.size(); naid++ ) {
									sb.append( (String) valarr.get(naid) );
									if ( naid < valarr.size()-1)
										sb.append( DataExpression.DELIM_NA_STRING_SEP );
								}
								naStrings = sb.toString();
							}
							else if ( val instanceof JSONObject ){
								JSONObject valJsonObject = (JSONObject)val;
								naStrings = valJsonObject.toString();
							}
							else {
								throw new ParseException("Type of value " + val
									+ " from metadata not recognized by parser.");
							}
							StringIdentifier sid = new StringIdentifier(naStrings, this);
							varParams.remove(key.toString());
							addVarParam(key.toString(), sid, varParams);
						}
						else {
							// by default, treat a parameter as a string
							addVarParam(key.toString(), strId, varParams);
						}
					}
				}
			}
		}
		return varParams;


//		//parse json meta data
//		long rows = jmtd.getLong(DataExpression.READROWPARAM);
//		long cols = jmtd.getLong(DataExpression.READCOLPARAM);
//		int blen = jmtd.containsKey(DataExpression.ROWBLOCKCOUNTPARAM)?
//			jmtd.getInt(DataExpression.ROWBLOCKCOUNTPARAM) : -1;
//		long nnz = jmtd.containsKey(DataExpression.READNNZPARAM)?
//			jmtd.getLong(DataExpression.READNNZPARAM) : -1;
//		String format = jmtd.getString(DataExpression.FORMAT_TYPE);
//		Types.FileFormat fmt = Types.FileFormat.safeValueOf(format);
	}

	public Object getParam(String key) {
		try {
			if(_metaObj.containsKey(key))
				return _metaObj.get(key);
		}
		catch(JSONException e) {
			e.printStackTrace();
		}
		return null;
	}

	public boolean containsParam(String key) { return _metaObj.containsKey(key); }

	public void addVarParam(String name, Expression value, HashMap<String, Expression> varParams) {
		if (DMLScript.VALIDATOR_IGNORE_ISSUES && (value == null)) {
			return;
		}
		varParams.put(name, value);

		// if required, initialize values
		setFilename(value.getFilename());
		if (getBeginLine() == 0) setBeginLine(value.getBeginLine());
		if (getBeginColumn() == 0) setBeginColumn(value.getBeginColumn());
		if (getEndLine() == 0) setEndLine(value.getEndLine());
		if (getEndColumn() == 0) setEndColumn(value.getEndColumn());
		if (getText() == null) setText(value.getText());
	}

}
