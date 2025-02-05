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

package org.apache.sysds.parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import static org.apache.sysds.runtime.instructions.fed.InitFEDInstruction.FED_FRAME_IDENTIFIER;
import static org.apache.sysds.runtime.instructions.fed.InitFEDInstruction.FED_MATRIX_IDENTIFIER;
import org.antlr.v4.runtime.ParserRuleContext;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.parser.LanguageException.LanguageErrorCodes;
import org.apache.sysds.parser.dml.CustomErrorListener;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.io.FileFormatPropertiesMM;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class DataExpression extends DataIdentifier
{
	private static final Log LOG = LogFactory.getLog(DataExpression.class.getName());

	public static final String RAND_DIMS = "dims";

	public static final String RAND_ROWS = "rows";
	public static final String RAND_COLS = "cols";
	public static final String RAND_MIN = "min";
	public static final String RAND_MAX = "max";
	public static final String RAND_SPARSITY = "sparsity";
	public static final String RAND_SEED = "seed";
	public static final String RAND_PDF = "pdf";
	public static final String RAND_LAMBDA = "lambda";
	
	public static final String RAND_PDF_UNIFORM = "uniform";
	
	public static final String RAND_BY_ROW = "byrow";
	public static final String RAND_DIMNAMES = "dimnames";
	public static final String RAND_DATA = "data";
	
	public static final String IO_FILENAME = "iofilename";
	public static final String READROWPARAM = "rows";
	public static final String READCOLPARAM = "cols";
	public static final String READNNZPARAM = "nnz";
	
	public static final String SQL_CONN = "conn";
	public static final String SQL_USER = "user";
	public static final String SQL_PASS = "password";
	public static final String SQL_QUERY = "query";
	
	public static final String FED_ADDRESSES = "addresses";
	public static final String FED_RANGES = "ranges";
	public static final String FED_TYPE = "type";
	public static final String FED_LOCAL_OBJECT = "local_matrix";
	
	public static final String FORMAT_TYPE = "format";
	
	public static final String ROWBLOCKCOUNTPARAM = "rows_in_block";
	public static final String COLUMNBLOCKCOUNTPARAM = "cols_in_block";
	public static final String DATATYPEPARAM = "data_type";
	public static final String VALUETYPEPARAM = "value_type";
	public static final String DESCRIPTIONPARAM = "description";
	public static final String AUTHORPARAM = "author";
	public static final String SCHEMAPARAM = "schema";
	public static final String CREATEDPARAM = "created";

	public static final String PRIVACY = "privacy";
	public static final String FINE_GRAINED_PRIVACY = "fine_grained_privacy";

	// Parameter names relevant to reading/writing delimited/csv files
	public static final String DELIM_DELIMITER = "sep";
	public static final String DELIM_HAS_HEADER_ROW = "header";
	public static final String DELIM_FILL = "fill";
	public static final String DELIM_FILL_VALUE = "default";
	//public static final String DELIM_RECODE = "recode";
	public static final String DELIM_NA_STRINGS = "naStrings";
	public static final String DELIM_NA_STRING_SEP = "\u00b7";
	// Parameter names relevant to reading/writing delimited index/libsvmv files
	public static final String LIBSVM_INDEX_DELIM = "indSep";

	// Parameter names relevant to reading/writing dataset name/hdf5 files
	public static final String HDF5_DATASET_NAME = "dataset";
	
	public static final String DELIM_SPARSE = "sparse";  // applicable only for write
	
	public static final Set<String> RAND_VALID_PARAM_NAMES = new HashSet<>(
		Arrays.asList(RAND_ROWS, RAND_COLS, RAND_DIMS,
			RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, RAND_PDF, RAND_LAMBDA));
	
	public static final Set<String> RESHAPE_VALID_PARAM_NAMES = new HashSet<>(
		Arrays.asList(RAND_BY_ROW, RAND_DIMNAMES, RAND_DATA, RAND_ROWS, RAND_COLS, RAND_DIMS));

	public static final Set<String> FRAME_VALID_PARAM_NAMES = new HashSet<>(
		Arrays.asList(SCHEMAPARAM, RAND_DATA, RAND_ROWS, RAND_COLS));

	public static final Set<String> SQL_VALID_PARAM_NAMES = new HashSet<>(
		Arrays.asList(SQL_CONN, SQL_USER, SQL_PASS, SQL_QUERY));
	
	public static final Set<String> FEDERATED_VALID_PARAM_NAMES = new HashSet<>(
		Arrays.asList(FED_ADDRESSES, FED_RANGES, FED_TYPE, FED_LOCAL_OBJECT));

	/** Valid parameter names in metadata file */
	public static final Set<String> READ_VALID_MTD_PARAM_NAMES =new HashSet<>(
		Arrays.asList(IO_FILENAME, READROWPARAM, READCOLPARAM, READNNZPARAM,
			FORMAT_TYPE, ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, DATATYPEPARAM,
			VALUETYPEPARAM, SCHEMAPARAM, DESCRIPTIONPARAM, AUTHORPARAM, CREATEDPARAM,
			// Parameters related to delimited/csv files.
			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW, DELIM_NA_STRINGS,
			// Parameters related to delimited/libsvm files.
			LIBSVM_INDEX_DELIM,
			//Parameters related to dataset name/HDF4 files.
			HDF5_DATASET_NAME,
			// Parameters related to privacy
			PRIVACY, FINE_GRAINED_PRIVACY));

	/** Valid parameter names in arguments to read instruction */
	public static final Set<String> READ_VALID_PARAM_NAMES = new HashSet<>(
		Arrays.asList(IO_FILENAME, READROWPARAM, READCOLPARAM, FORMAT_TYPE, DATATYPEPARAM,
			VALUETYPEPARAM, SCHEMAPARAM, ROWBLOCKCOUNTPARAM, COLUMNBLOCKCOUNTPARAM, READNNZPARAM,
			// Parameters related to delimited/csv files.
			DELIM_FILL_VALUE, DELIM_DELIMITER, DELIM_FILL, DELIM_HAS_HEADER_ROW, DELIM_NA_STRINGS,
			// Parameters related to delimited/libsvm files.
			LIBSVM_INDEX_DELIM,
			//Parameters related to dataset name/HDF4 files.
			HDF5_DATASET_NAME));
	
	/* Default Values for delimited (CSV/LIBSVM) files */
	public static final String  DEFAULT_DELIM_DELIMITER = ",";
	public static final boolean DEFAULT_DELIM_HAS_HEADER_ROW = false;
	public static final boolean DEFAULT_DELIM_FILL = true;
	public static final double  DEFAULT_DELIM_FILL_VALUE = 0.0;
	public static final boolean DEFAULT_DELIM_SPARSE = false;
	public static final String  DEFAULT_NA_STRINGS = "";
	public static final String  DEFAULT_SCHEMAPARAM = "NULL";
	public static final String DEFAULT_LIBSVM_INDEX_DELIM = ":";
	private static Map<String, Object> csvDefaults;
	static {
		csvDefaults = new HashMap<>();
		csvDefaults.put(DELIM_DELIMITER, DEFAULT_DELIM_DELIMITER);
		csvDefaults.put(DELIM_HAS_HEADER_ROW, DEFAULT_DELIM_HAS_HEADER_ROW);
		csvDefaults.put(DELIM_FILL, DEFAULT_DELIM_FILL);
		csvDefaults.put(DELIM_FILL_VALUE, DEFAULT_DELIM_FILL_VALUE);
		csvDefaults.put(DELIM_SPARSE, DEFAULT_DELIM_SPARSE);
		csvDefaults.put(DELIM_NA_STRINGS, DEFAULT_NA_STRINGS);
		csvDefaults.put(SCHEMAPARAM, DEFAULT_SCHEMAPARAM);
		csvDefaults.put(LIBSVM_INDEX_DELIM, DEFAULT_LIBSVM_INDEX_DELIM);
	}
	
	private DataOp _opcode;
	private HashMap<String, Expression> _varParams;
	private boolean _strInit = false; //string initialize
	private boolean _checkMetadata = true; // local skip meta data reads

	public DataExpression(){
		//do nothing
	}

	
	public void setCheckMetadata(boolean checkMetadata) {
		_checkMetadata = checkMetadata;
	}

	public static DataExpression getDataExpression(ParserRuleContext ctx, String functionName,
			ArrayList<ParameterExpression> passedParamExprs, String filename, CustomErrorListener errorListener) {
		ParseInfo pi = ParseInfo.ctxAndFilenameToParseInfo(ctx, filename);
		return getDataExpression(functionName, passedParamExprs, pi, errorListener);
	}

	public static DataExpression getDataExpression(String functionName, ArrayList<ParameterExpression> passedParamExprs,
			ParseInfo parseInfo, CustomErrorListener errorListener) {
		if (functionName == null || passedParamExprs == null)
			return null;
		if( LOG.isDebugEnabled() ) {
			LOG.debug("getDataExpression: " + functionName + " " 
				+ passedParamExprs + " " + parseInfo + " " + errorListener);
		}
		// check if the function name is built-in function
		// (assign built-in function op if function is built-in)
		DataExpression dataExpr = null;
		if (functionName.equals("read") || functionName.equals("readMM") || functionName.equals("read.csv"))
			dataExpr = processReadDataExpression(functionName, passedParamExprs, errorListener, parseInfo);
		else if (functionName.equalsIgnoreCase("rand"))
			dataExpr = processRandDataExpression(functionName, passedParamExprs, errorListener, parseInfo);
		else if (functionName.equals("matrix"))
			dataExpr = processMatrixExpression(functionName, passedParamExprs, errorListener, parseInfo);
		else if (functionName.equals("frame"))
			dataExpr = processFrameExpression(functionName, passedParamExprs, errorListener, parseInfo);
		else if (functionName.equals("tensor"))
			dataExpr = processTensorExpression(functionName, passedParamExprs, errorListener, parseInfo);
		else if (functionName.equals("sql"))
			dataExpr = processSQLExpression(functionName, passedParamExprs, errorListener, parseInfo);
		else if (functionName.equals("federated"))
			dataExpr = processFederatedExpression(functionName, passedParamExprs, errorListener, parseInfo);
		
		if (dataExpr != null)
			dataExpr.setParseInfo(parseInfo);
		return dataExpr;
	}
	
	private static DataExpression processReadDataExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.READ, new HashMap<>(), parseInfo);
		if (functionName.equals("readMM"))
			dataExpr.addVarParam(DataExpression.FORMAT_TYPE,
				new StringIdentifier(FileFormat.MM.toString(), parseInfo));

		if (functionName.equals("read.csv"))
			dataExpr.addVarParam(DataExpression.FORMAT_TYPE,
				new StringIdentifier(FileFormat.CSV.toString(), parseInfo));

		if (functionName.equals("read.libsvm"))
			dataExpr.addVarParam(DataExpression.FORMAT_TYPE,
				new StringIdentifier(FileFormat.LIBSVM.toString(), parseInfo));

		// validate the filename is the first parameter
		if (passedParamExprs.size() < 1){
			errorListener.validationError(parseInfo, "read method must have at least filename parameter");
			return null;
		}
		
		ParameterExpression pexpr = (passedParamExprs.size() == 0) ? null : passedParamExprs.get(0);
		
		if ( (pexpr != null) &&  (!(pexpr.getName() == null) || (pexpr.getName() != null && pexpr.getName().equalsIgnoreCase(DataExpression.IO_FILENAME)))){
			errorListener.validationError(parseInfo, "first parameter to read statement must be filename");
			return null;
		} else if( pexpr != null ){
			dataExpr.addVarParam(DataExpression.IO_FILENAME, pexpr.getExpr());
		}
		
		// validate all parameters are added only once and valid name
		for (int i = 1; i < passedParamExprs.size(); i++){
			String currName = passedParamExprs.get(i).getName();
			Expression currExpr = passedParamExprs.get(i).getExpr();
			
			if (dataExpr.getVarParam(currName) != null){
				errorListener.validationError(parseInfo, "attempted to add IOStatement parameter " + currName + " more than once");
				return null;
			}
			// verify parameter names for read function
			boolean isValidName = READ_VALID_PARAM_NAMES.contains(currName);

			if (!isValidName){
				errorListener.validationError(parseInfo, "attempted to add invalid read statement parameter " + currName);
				return null;
			}
			dataExpr.addVarParam(currName, currExpr);
		}
		
		return dataExpr;
	}
	
	private static DataExpression processRandDataExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.RAND, new HashMap<>(), parseInfo);
		
		for (ParameterExpression currExpr : passedParamExprs){
			String pname = currExpr.getName();
			Expression pexpr = currExpr.getExpr();
			if (pname == null){
				errorListener.validationError(parseInfo, "for rand statement, all arguments must be named parameters");
				return null;
			}
			dataExpr.addRandExprParam(pname, pexpr);
		}
		dataExpr.setRandDefault();
		return dataExpr;
	}
	
	private static DataExpression processMatrixExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.MATRIX, new HashMap<>(), parseInfo);
		int namedParamCount = (int) passedParamExprs.stream().filter(p -> p.getName()!=null).count();
		int unnamedParamCount = passedParamExprs.size() - namedParamCount;
		
		// check whether named or unnamed parameters are used
		if (passedParamExprs.size() < 3){
			errorListener.validationError(parseInfo, "for matrix statement, must specify at least 3 arguments: data, rows, cols");
			return null;
		}
		
		if (unnamedParamCount > 1){
			if (namedParamCount > 0) {
				errorListener.validationError(parseInfo, "for matrix statement, cannot mix named and unnamed parameters");
				return null;
			}
			if (unnamedParamCount < 3) {
				errorListener.validationError(parseInfo, "for matrix statement, must specify at least 3 arguments: data, rows, cols");
				return null;
			}

			// assume: data, rows, cols, [byRow], [dimNames]
			dataExpr.addMatrixExprParam(DataExpression.RAND_DATA,passedParamExprs.get(0).getExpr());
			dataExpr.addMatrixExprParam(DataExpression.RAND_ROWS,passedParamExprs.get(1).getExpr());
			dataExpr.addMatrixExprParam(DataExpression.RAND_COLS,passedParamExprs.get(2).getExpr());
			
			if (unnamedParamCount >= 4)
				dataExpr.addMatrixExprParam(DataExpression.RAND_BY_ROW,passedParamExprs.get(3).getExpr());
			if (unnamedParamCount == 5)
				dataExpr.addMatrixExprParam(DataExpression.RAND_DIMNAMES,passedParamExprs.get(4).getExpr());
			if (unnamedParamCount > 5) {
				errorListener.validationError(parseInfo, "for matrix statement, at most 5 arguments supported: data, rows, cols, byrow, dimname");
				return null;
			}
		}
		else {
			// handle first parameter, which is data and may be unnamed
			ParameterExpression firstParam = passedParamExprs.get(0);
			if (firstParam.getName() != null && !firstParam.getName().equals(DataExpression.RAND_DATA)){
				errorListener.validationError(parseInfo, "matrix method must have data parameter as first parameter or unnamed parameter");
				return null;
			} else {
				dataExpr.addMatrixExprParam(DataExpression.RAND_DATA, passedParamExprs.get(0).getExpr());
			}
			
			for (int i=1; i<passedParamExprs.size(); i++){
				if (passedParamExprs.get(i).getName() == null){
					errorListener.validationError(parseInfo, "for matrix statement, cannot mix named and unnamed parameters, only data parameter can be unnamed");
					return null;
				} else {
					dataExpr.addMatrixExprParam(passedParamExprs.get(i).getName(), passedParamExprs.get(i).getExpr());
				}
			}
		}
		dataExpr.setMatrixDefault();
		return dataExpr;
	}
	
	private static DataExpression processFrameExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.FRAME, new HashMap<>(), parseInfo);
		int namedParamCount = (int) passedParamExprs.stream().filter(p -> p.getName()!=null).count();
		int unnamedParamCount = passedParamExprs.size() - namedParamCount;

		// check whether named or unnamed parameters are used
		if (passedParamExprs.size() < 3) { // it will generate a frame with string schema
			errorListener.validationError(parseInfo, "for frame statement, must specify at least 3 arguments: data, rows and cols");
			return null;
		}

		if (unnamedParamCount > 1) {
			if (namedParamCount > 0) {
				errorListener.validationError(parseInfo, "for frame statement, cannot mix named and unnamed parameters");
				return null;
			}
			if (unnamedParamCount < 3) {
				errorListener.validationError(parseInfo, "for frame statement, must specify at least 3 arguments: rows, cols");
				return null;
			}
			// assume: data, rows, cols, [Schema]
			dataExpr.addFrameExprParam(DataExpression.RAND_DATA, passedParamExprs.get(0).getExpr());
			dataExpr.addFrameExprParam(DataExpression.RAND_ROWS, passedParamExprs.get(1).getExpr());
			dataExpr.addFrameExprParam(DataExpression.RAND_COLS, passedParamExprs.get(2).getExpr());

			if (unnamedParamCount == 3)
				dataExpr.addFrameExprParam(DataExpression.SCHEMAPARAM, passedParamExprs.get(3).getExpr());
			if (unnamedParamCount > 3) {
				errorListener.validationError(parseInfo, "for frame  statement, at most 4 arguments supported: data, rows, cols, schema");
				return null;
			}
		}
		else {
			// handle first parameter, which is data and may be unnamed
			ParameterExpression firstParam = passedParamExprs.get(0);
			if (firstParam.getName() != null && !firstParam.getName().equals(DataExpression.RAND_DATA)){
				errorListener.validationError(parseInfo, "frame method must have data parameter as first parameter or unnamed parameter");
				return null;
			}
			else {
				dataExpr.addFrameExprParam(DataExpression.RAND_DATA, passedParamExprs.get(0).getExpr());
			}

			for (int i=1; i<passedParamExprs.size(); i++){
				if (passedParamExprs.get(i).getName() == null){
					errorListener.validationError(parseInfo, "for frame statement, cannot mix named and unnamed parameters, only data parameter can be unnamed");
					return null;
				} else {
					dataExpr.addFrameExprParam(passedParamExprs.get(i).getName(), passedParamExprs.get(i).getExpr());
				}
			}
		}
		dataExpr.setFrameDefault();
		return dataExpr;
	}
	
	private static DataExpression processTensorExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.TENSOR, new HashMap<>(), parseInfo);
		int namedParamCount = (int) passedParamExprs.stream().filter(p -> p.getName()!=null).count();
		int unnamedParamCount = passedParamExprs.size() - namedParamCount;

		// check whether named or unnamed parameters are used
		if (passedParamExprs.size() < 2){
			errorListener.validationError(parseInfo, "for tensor statement, must specify at least 2 arguments: data, dims[]");
			return null;
		}
		if (unnamedParamCount > 1){
			if (namedParamCount > 0) {
				errorListener.validationError(parseInfo, "for tensor statement, cannot mix named and unnamed parameters");
				return null;
			}

			// assume: data, dims[], [byRow], [dimNames]
			dataExpr.addTensorExprParam(DataExpression.RAND_DATA,passedParamExprs.get(0).getExpr());
			dataExpr.addTensorExprParam(DataExpression.RAND_DIMS,passedParamExprs.get(1).getExpr());

			if (unnamedParamCount >= 3)
				// TODO use byRow parameter
				dataExpr.addTensorExprParam(DataExpression.RAND_BY_ROW,passedParamExprs.get(2).getExpr());
			if (unnamedParamCount == 4)
				dataExpr.addTensorExprParam(DataExpression.RAND_DIMNAMES,passedParamExprs.get(3).getExpr());
			if (unnamedParamCount > 4) {
				errorListener.validationError(parseInfo, "for tensor statement, at most 4 arguments supported: data, dims, byrow, dimname");
				return null;
			}
		}
		else {
			// handle first parameter, which is data and may be unnamed
			ParameterExpression firstParam = passedParamExprs.get(0);
			if (firstParam.getName() != null && !firstParam.getName().equals(DataExpression.RAND_DATA)){
				errorListener.validationError(parseInfo, "tensor method must have data parameter as first parameter or unnamed parameter");
				return null;
			}
			else {
				dataExpr.addTensorExprParam(DataExpression.RAND_DATA, passedParamExprs.get(0).getExpr());
			}

			for (int i=1; i<passedParamExprs.size(); i++){
				if (passedParamExprs.get(i).getName() == null){
					errorListener.validationError(parseInfo, "for tensor statement, cannot mix named and unnamed parameters, only data parameter can be unnamed");
					return null;
				}
				else {
					dataExpr.addTensorExprParam(passedParamExprs.get(i).getName(), passedParamExprs.get(i).getExpr());
				}
			}
		}
		dataExpr.setTensorDefault();
		return dataExpr;
	}
	
	private static DataExpression processSQLExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.SQL, new HashMap<>(), parseInfo);
		int namedParamCount = (int) passedParamExprs.stream().filter(p -> p.getName()!=null).count();
		int unnamedParamCount = passedParamExprs.size() - namedParamCount;
		
		// check whether named or unnamed parameters are used
		if (passedParamExprs.size() < 2){
			errorListener.validationError(parseInfo, "for sql statement, must specify at least 2 arguments: conn, query");
			return null;
		}
		if (unnamedParamCount > 0){
			if (namedParamCount > 0) {
				errorListener.validationError(parseInfo, "for sql statement, cannot mix named and unnamed parameters");
				return null;
			}
			if (unnamedParamCount == 2 || unnamedParamCount == 4 ) {
				// assume: conn, query, [password, query]
				dataExpr.addSqlExprParam(DataExpression.SQL_CONN, passedParamExprs.get(0).getExpr());
				dataExpr.addSqlExprParam(DataExpression.SQL_QUERY, passedParamExprs.get(1).getExpr());
				if (unnamedParamCount == 4) {
					dataExpr.addSqlExprParam(DataExpression.SQL_PASS, passedParamExprs.get(2).getExpr());
					dataExpr.addSqlExprParam(DataExpression.SQL_QUERY, passedParamExprs.get(3).getExpr());
				}
			}
			else {
				errorListener.validationError(parseInfo, "for sql statement, "
					+ "at most 4 arguments supported: conn, user, password, query");
				return null;
			}
		}
		else {
			for (ParameterExpression passedParamExpr : passedParamExprs) {
				dataExpr.addSqlExprParam(passedParamExpr.getName(), passedParamExpr.getExpr());
			}
		}
		dataExpr.setSqlDefault();
		return dataExpr;
	}
	
	private static DataExpression processFederatedExpression(String functionName,
		List<ParameterExpression> passedParamExprs, CustomErrorListener errorListener, ParseInfo parseInfo)
	{
		DataExpression dataExpr = new DataExpression(DataOp.FEDERATED, new HashMap<>(), parseInfo);
		int namedParamCount = (int) passedParamExprs.stream().filter(p -> p.getName()!=null).count();
		int unnamedParamCount = passedParamExprs.size() - namedParamCount;

		if(passedParamExprs.size() < 2) {
			errorListener.validationError(parseInfo,
				"for federated statement, must specify at least 2 arguments: addresses, ranges");
			return null;
		}
		if(unnamedParamCount > 0) {
			if(namedParamCount > 0) {
				errorListener.validationError(parseInfo,
					"for federated statement, cannot mix named and unnamed parameters");
				return null;
			}
			if(unnamedParamCount == 2) {
				// first parameter addresses second are the ranges (type defaults to Matrix)
				ParameterExpression param = passedParamExprs.get(0);
				dataExpr.addFederatedExprParam(DataExpression.FED_ADDRESSES, param.getExpr());
				param = passedParamExprs.get(1);
				dataExpr.addFederatedExprParam(DataExpression.FED_RANGES, param.getExpr());
			}
			else if(unnamedParamCount == 3) {
				ParameterExpression param = passedParamExprs.get(0);
				dataExpr.addFederatedExprParam(DataExpression.FED_ADDRESSES, param.getExpr());
				param = passedParamExprs.get(1);
				dataExpr.addFederatedExprParam(DataExpression.FED_RANGES, param.getExpr());
				param = passedParamExprs.get(2);
				dataExpr.addFederatedExprParam(DataExpression.FED_TYPE, param.getExpr());
			}
			else if(unnamedParamCount == 4) {
				ParameterExpression param = passedParamExprs.get(0);
				dataExpr.addFederatedExprParam(DataExpression.FED_LOCAL_OBJECT, param.getExpr());
				param = passedParamExprs.get(1);
				dataExpr.addFederatedExprParam(DataExpression.FED_ADDRESSES, param.getExpr());
				param = passedParamExprs.get(2);
				dataExpr.addFederatedExprParam(DataExpression.FED_RANGES, param.getExpr());
				param = passedParamExprs.get(3);
				dataExpr.addFederatedExprParam(DataExpression.FED_TYPE, param.getExpr());
			}
			else {
				errorListener.validationError(parseInfo,
					"for federated statement, at most 3 arguments are supported: addresses, ranges, type");
			}
		}
		else {
			for (ParameterExpression passedParamExpr : passedParamExprs) {
				dataExpr.addFederatedExprParam(passedParamExpr.getName(), passedParamExpr.getExpr());
			}
		}
		dataExpr.setFederatedDefault();
		return dataExpr;
	}
	
	public void addRandExprParam(String paramName, Expression paramValue)
	{
		if (DMLScript.VALIDATOR_IGNORE_ISSUES && (paramValue == null)) {
			return;
		}
		// check name is valid
		boolean found = RAND_VALID_PARAM_NAMES.contains(paramName);
		if (!found){
			raiseValidateError("unexpected parameter \"" + paramName +
					"\". Legal parameters for Rand statement are " 
					+ "(capitalization-sensitive): " 	+ RAND_ROWS 	
					+ ", " + RAND_COLS		+ ", " + RAND_MIN + ", " + RAND_MAX  	
					+ ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", " + RAND_PDF + ", " + RAND_LAMBDA);
		}
		if (getVarParam(paramName) != null){
			raiseValidateError("attempted to add Rand statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier) {
			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
		} else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier) {
			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
		}
		// add the parameter to expression list
		paramValue.setParseInfo(this);
		addVarParam(paramName,paramValue);
		
	}
	
	public void addMatrixExprParam(String paramName, Expression paramValue) 
	{
		// check name is valid
		boolean found = RESHAPE_VALID_PARAM_NAMES.contains(paramName);
		
		if (!found){
			raiseValidateError("unexpected parameter \"" + paramName +
					"\". Legal parameters for  matrix statement are " 
					+ "(capitalization-sensitive): " + RAND_DATA + ", " + RAND_ROWS
					+ ", " + RAND_COLS + ", " + RAND_BY_ROW);
		}
		if (getVarParam(paramName) != null) {
			raiseValidateError("attempted to add matrix statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		if (paramName.equals(RAND_ROWS) && paramValue instanceof DoubleIdentifier) {
			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
		} else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier) {
			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
		}

		// add the parameter to expression list
		paramValue.setParseInfo(this);
		addVarParam(paramName,paramValue);
	}
	public void addFrameExprParam(String paramName, Expression paramValue)
	{
		// check name is valid
		boolean found = FRAME_VALID_PARAM_NAMES.contains(paramName);

		if (!found){
			raiseValidateError("unexpected parameter \"" + paramName +
				"\". Legal parameters for  frame statement are "
				+ "(capitalization-sensitive): " + RAND_DATA + ", " + RAND_ROWS
				+ ", " + RAND_COLS + ", " + SCHEMAPARAM);
		}
		if (getVarParam(paramName) != null) {
			raiseValidateError("attempted to add frame statement parameter " + paramValue + " more than once");
		}
//		TODO convert double Matrix to String Frame
		// Process the case where user provides double values to rows or cols
//		if (paramName.equals(RAND_ROWS) && paramValue instanceof StringIdentifier) {
//			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
//		} else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier) {
//			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
//		}

		// add the parameter to expression list
		paramValue.setParseInfo(this);
		addVarParam(paramName,paramValue);
	}

	public void addTensorExprParam(String paramName, Expression paramValue)
	{
		// check name is valid
		boolean found = RESHAPE_VALID_PARAM_NAMES.contains(paramName);

		if (!found){
			raiseValidateError("unexpected parameter \"" + paramName + "\". Legal parameters for tensor statement are "
					+ "(capitalization-sensitive): " + RAND_DATA + ", " + RAND_DIMS +
					", " + RAND_BY_ROW + ", " + RAND_DIMNAMES);
		}
		if (getVarParam(paramName) != null) {
			raiseValidateError("attempted to add tensor statement parameter " + paramValue + " more than once");
		}
		// Process the case where user provides double values to rows or cols
		// TODO convert double Matrix to long Matrix
		/*if (paramName.equals(RAND_DIMS) && paramValue instanceof DoubleIdentifier) {
			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
		} else if (paramName.equals(RAND_COLS) && paramValue instanceof DoubleIdentifier) {
			paramValue = new IntIdentifier((long) ((DoubleIdentifier) paramValue).getValue(), this);
		}*/

		// add the parameter to expression list
		paramValue.setParseInfo(this);
		addVarParam(paramName,paramValue);
	}
	
	public void addSqlExprParam(String paramName, Expression paramValue)
	{
		// check name is valid
		boolean found = SQL_VALID_PARAM_NAMES.contains(paramName);
		
		if (!found){
			raiseValidateError("unexpected parameter \"" + paramName + "\". Legal parameters for sql statement are "
					+ "(capitalization-sensitive): " + SQL_CONN + ", " + SQL_USER + ", " + SQL_PASS + ", " + SQL_QUERY);
		}
		if (getVarParam(paramName) != null) {
			raiseValidateError("attempted to add sql statement parameter " + paramValue + " more than once");
		}
		
		// add the parameter to expression list
		paramValue.setParseInfo(this);
		addVarParam(paramName,paramValue);
	}
	
	public void addFederatedExprParam(String paramName, Expression paramValue) {
		// check name is valid
		boolean found = FEDERATED_VALID_PARAM_NAMES.contains(paramName);
		
		if (!found)
			raiseValidateError("unexpected parameter \"" + paramName + "\". Legal parameters for federated statement are "
				+ "(capitalization-sensitive): " + FED_ADDRESSES + ", " + FED_RANGES + ", " + FED_TYPE);
		if (getVarParam(paramName) != null)
			raiseValidateError("attempted to add federated statement parameter " + paramValue + " more than once");
		
		// add the parameter to expression list
		paramValue.setParseInfo(this);
		addVarParam(paramName,paramValue);
	}
	
	public DataExpression(DataOp op, HashMap<String, Expression> varParams, ParseInfo parseInfo) {
		_opcode = op;
		_varParams = varParams;
		setParseInfo(parseInfo);
	}

	public DataExpression(ParserRuleContext ctx, DataOp op, HashMap<String,Expression> varParams, 
			String filename) {
		_opcode = op;
		_varParams = varParams;
		setCtxValuesAndFilename(ctx, filename);
	}

	@Override
	public Expression rewriteExpression(String prefix) {
		HashMap<String,Expression> newVarParams = new HashMap<>();
		for( Entry<String, Expression> e : _varParams.entrySet() ){
			String key = e.getKey();
			Expression newExpr = e.getValue().rewriteExpression(prefix);
			newVarParams.put(key, newExpr);
		}
		DataExpression retVal = new DataExpression(_opcode, newVarParams, this);
		retVal._strInit = this._strInit;
		
		return retVal;
	}

	/**
	 * By default we use rowwise matrix reshape according to our internal dense/sparse matrix representations.
	 * ByRow specifies both input and output orientation. Note that this is different from R, where inputs are 
	 * always read by-column and the default for byRow is by-column as well.
	 */
	public void setMatrixDefault(){
		if (getVarParam(RAND_BY_ROW) == null)
			addVarParam(RAND_BY_ROW, new BooleanIdentifier(true, this));
	}

	public void setFrameDefault(){
		if(getVarParam(RAND_DATA) == null)
			addVarParam(RAND_DATA, new StringIdentifier(null, this));
		if (getVarParam(SCHEMAPARAM) == null)
			addVarParam(SCHEMAPARAM, new StringIdentifier(DEFAULT_SCHEMAPARAM, this));
	}

	public void setTensorDefault(){
		if (getVarParam(RAND_BY_ROW) == null)
			addVarParam(RAND_BY_ROW, new BooleanIdentifier(true, this));
	}
	
	public void setFederatedDefault(){
		if (getVarParam(FED_TYPE) == null)
			addVarParam(FED_TYPE, new StringIdentifier(FED_MATRIX_IDENTIFIER, this));
	}
	
	private void setSqlDefault() {
		if (getVarParam(SQL_USER) == null)
			addVarParam(SQL_USER, new StringIdentifier("", this));
		if (getVarParam(SQL_PASS) == null)
			addVarParam(SQL_PASS, new StringIdentifier("", this));
	}
	
	
	public void setRandDefault() {
		if (getVarParam(RAND_DIMS) == null) {
			if( getVarParam(RAND_ROWS) == null ) {
				IntIdentifier id = new IntIdentifier(1L, this);
				addVarParam(RAND_ROWS, id);
			}
			if( getVarParam(RAND_COLS) == null ) {
				IntIdentifier id = new IntIdentifier(1L, this);
				addVarParam(RAND_COLS, id);
			}
		}
		if (getVarParam(RAND_MIN) == null) {
			DoubleIdentifier id = new DoubleIdentifier(0.0, this);
			addVarParam(RAND_MIN, id);
		}
		if (getVarParam(RAND_MAX) == null) {
			DoubleIdentifier id = new DoubleIdentifier(1.0, this);
			addVarParam(RAND_MAX, id);
		}
		if (getVarParam(RAND_SPARSITY) == null) {
			DoubleIdentifier id = new DoubleIdentifier(1.0, this);
			addVarParam(RAND_SPARSITY, id);
		}
		if (getVarParam(RAND_SEED) == null) {
			IntIdentifier id = new IntIdentifier(DataGenOp.UNSPECIFIED_SEED, this);
			addVarParam(RAND_SEED, id);
		}
		if (getVarParam(RAND_PDF) == null) {
			StringIdentifier id = new StringIdentifier(RAND_PDF_UNIFORM, this);
			addVarParam(RAND_PDF, id);
		}
		if (getVarParam(RAND_LAMBDA) == null) {
			DoubleIdentifier id = new DoubleIdentifier(1.0, this);
			addVarParam(RAND_LAMBDA, id);
		}
	}

	public void setOpCode(DataOp op) {
		_opcode = op;
	}
	
	public DataOp getOpCode() {
		return _opcode;
	}
	
	public HashMap<String,Expression> getVarParams() {
		return _varParams;
	}
	
	public void setVarParams(HashMap<String, Expression> varParams) {
		_varParams = varParams;
	}
	
	public Expression getVarParam(String name) {
		return _varParams.get(name);
	}

	public void addVarParam(String name, Expression value){
		if (DMLScript.VALIDATOR_IGNORE_ISSUES && (value == null)) {
			return;
		}
		_varParams.put(name, value);
		
		// if required, initialize values
		setFilename(value.getFilename());
		if (getBeginLine() == 0) setBeginLine(value.getBeginLine());
		if (getBeginColumn() == 0) setBeginColumn(value.getBeginColumn());
		if (getEndLine() == 0) setEndLine(value.getEndLine());
		if (getEndColumn() == 0) setEndColumn(value.getEndColumn());
		if (getText() == null) setText(value.getText());
	}
	
	public void removeVarParam(String name) {
		_varParams.remove(name);
	}
	
	public void removeVarParam(String... names) {
		for( String name : names )
			removeVarParam(name);
	}
	
	private String getInputFileName(HashMap<String, ConstIdentifier> currConstVars, boolean conditional) {
		String filename = null;
		
		Expression fileNameExpr = getVarParam(IO_FILENAME);
		if (fileNameExpr instanceof ConstIdentifier){
			return fileNameExpr.toString();
		}
		else if (fileNameExpr instanceof BinaryExpression) {
			BinaryExpression expr = (BinaryExpression)fileNameExpr;
			Expression.BinaryOp op = expr.getOpCode();
			switch (op){
			case PLUS:
				filename = "";
				filename = fileNameCat(expr, currConstVars, filename, conditional);
				// Since we have computed the value of filename, we update
				// varParams with a const string value
				StringIdentifier fileString = new StringIdentifier(filename, this);
				removeVarParam(IO_FILENAME);
				addVarParam(IO_FILENAME, fileString);
				break;
			default:
				raiseValidateError("for read method, parameter " + IO_FILENAME + " can only be const string concatenations. ", conditional);
			}
		}
		else {
			raiseValidateError("for read method, parameter " + IO_FILENAME + " can only be a const string or const string concatenations. ", conditional);
		}
		
		return filename;
	}
	
	public static String getMTDFileName(String inputFileName) {
		return inputFileName + ".mtd";
	}
	
	/**
	 * Validate parse tree : Process Data Expression in an assignment
	 * statement
	 */
	@Override
	public void validateExpression(HashMap<String, DataIdentifier> ids, HashMap<String, ConstIdentifier> currConstVars, boolean conditional)
	{
		// validate all input parameters
		for ( Entry<String,Expression> e : getVarParams().entrySet() ) {
			String s = e.getKey();
			Expression inputParamExpr = e.getValue();
			
			if (inputParamExpr instanceof FunctionCallIdentifier) {
				raiseValidateError("UDF function call not supported as parameter to built-in function call", false,LanguageErrorCodes.INVALID_PARAMETERS);
			}
			inputParamExpr.validateExpression(ids, currConstVars, conditional);
			if (s != null && !s.equals(RAND_DATA) && !s.equals(RAND_DIMS) && !s.equals(FED_ADDRESSES) && !s.equals(FED_RANGES) && !s.equals(FED_LOCAL_OBJECT)
					&& !s.equals(DELIM_NA_STRINGS) && !s.equals(SCHEMAPARAM) && getVarParam(s).getOutput().getDataType() != DataType.SCALAR ) {
				raiseValidateError("Non-scalar data types are not supported for data expression.", conditional,LanguageErrorCodes.INVALID_PARAMETERS);
			}
		}
	
		//general data expression constant propagation
		performConstantPropagationRand( currConstVars );
		performConstantPropagationReadWrite( currConstVars );
		
		// check if data parameter of matrix is scalar or matrix -- if scalar, use Rand instead
		Expression dataParam1 = getVarParam(RAND_DATA);
		if (dataParam1 == null && (getOpCode().equals(DataOp.MATRIX) || getOpCode().equals(DataOp.TENSOR))){
			raiseValidateError("for matrix, frame or tensor, must defined data parameter", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
		}
		// We need to remember the operation if we replace the OpCode by rand so we have the correct output
		if (dataParam1!=null && dataParam1.getOutput()!=null && dataParam1.getOutput().getDataType() == DataType.SCALAR &&
				(_opcode == DataOp.MATRIX || _opcode == DataOp.TENSOR)/*&& dataParam instanceof ConstIdentifier*/ ){
			//MB: note we must not check for const identifiers here, because otherwise all matrix constructors with
			//variable input are routed to a reshape operation (but it works only on matrices and hence, crashes)
			
			// replace DataOp MATRIX with RAND -- Rand handles matrix generation for Scalar values
			// replace data parameter with min / max within Rand case below
			this.setOpCode(DataOp.RAND);
		}

		// IMPORTANT: for each operation, one must handle unnamed parameters
		
		switch (this.getOpCode()) {
		
		case READ:
			if (getVarParam(DATATYPEPARAM) != null && !(getVarParam(DATATYPEPARAM) instanceof StringIdentifier)){
				raiseValidateError("for read statement, parameter " + DATATYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.MATRIX_DATA_TYPE +", " + Statement.SCALAR_DATA_TYPE, conditional);
			}
			
			String dataTypeString = (getVarParam(DATATYPEPARAM) == null) ? null : getVarParam(DATATYPEPARAM).toString();
			
			// disallow certain parameters while reading a scalar
			if (dataTypeString != null && dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)){
				if ( getVarParam(READROWPARAM) != null
						|| getVarParam(READCOLPARAM) != null
						|| getVarParam(ROWBLOCKCOUNTPARAM) != null
						|| getVarParam(COLUMNBLOCKCOUNTPARAM) != null
						|| getVarParam(FORMAT_TYPE) != null
						|| getVarParam(DELIM_DELIMITER) != null	
						|| getVarParam(LIBSVM_INDEX_DELIM) != null
						|| getVarParam(DELIM_HAS_HEADER_ROW) != null
						|| getVarParam(DELIM_FILL) != null
						|| getVarParam(DELIM_FILL_VALUE) != null
						|| getVarParam(DELIM_NA_STRINGS) != null
						)
				{
					raiseValidateError("Invalid parameters in read statement of a scalar: " +
						toString() + ". Only " + VALUETYPEPARAM + " is allowed.", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
			}

			MetaDataAll configObj = new MetaDataAll();

			// Process expressions in input filename
			String inputFileName = getInputFileName(currConstVars, conditional);
			
			// Obtain and validate metadata filename
			String mtdFileName = getMTDFileName(inputFileName);

			// track whether should attempt to read MTD file or not
			boolean shouldReadMTD = _checkMetadata
				&& !(dataTypeString!= null && getVarParam(VALUETYPEPARAM) != null 
					&& dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE))
				&& (!ConfigurationManager.getCompilerConfigFlag(ConfigType.IGNORE_READ_WRITE_METADATA)
					|| HDFSTool.existsFileOnHDFS(mtdFileName)); // existing mtd file

			// Check for file existence (before metadata parsing for meaningful error messages)
			if( shouldReadMTD //skip check for jmlc/mlcontext
				&& !HDFSTool.existsFileOnHDFS(inputFileName)
				&& !ConfigurationManager.getCompilerConfigFlag(ConfigType.RESOURCE_OPTIMIZATION))
			{
				String fsext = InfrastructureAnalyzer.isLocalMode() ? "FS (local mode)" : "HDFS";
				raiseValidateError("Read input file does not exist on "+fsext+": " + 
					inputFileName, conditional);
			}

			// track whether format type has been inferred 
			boolean inferredFormatType = false;
			
			// get format type string
			String formatTypeString = (getVarParam(FORMAT_TYPE) == null) ?
				null : getVarParam(FORMAT_TYPE).toString();
			
			// check if file is matrix market format
			if (formatTypeString == null && shouldReadMTD){
				if ( MetaDataAll.checkHasMatrixMarketFormat(inputFileName, mtdFileName, conditional) ) {
					formatTypeString = FileFormat.MM.toString();
					addVarParam(FORMAT_TYPE, new StringIdentifier(formatTypeString, this));
					configObj.setFormatTypeString(formatTypeString);
					inferredFormatType = true;
					shouldReadMTD = false;
				}
			}

			// check if file is delimited format
			if (formatTypeString == null && shouldReadMTD ) {
				formatTypeString = MetaDataAll.checkHasDelimitedFormat(inputFileName, conditional);
				if (formatTypeString != null) {
					addVarParam(FORMAT_TYPE, new StringIdentifier(formatTypeString, this));
					configObj.setFormatTypeString(formatTypeString);
					inferredFormatType = true;
				}
			}
			
			if (formatTypeString != null && formatTypeString.equalsIgnoreCase(FileFormat.MM.toString())){
				/*
				 *  handle MATRIXMARKET_FORMAT_TYPE format
				 *
				 * 1) only allow IO_FILENAME as ONLY valid parameter
				 * 
				 * 2) open the file
				 *  A) verify header line (1st line) equals 
				 *  B) read and discard comment lines
				 *  C) get size information from sizing info line --- M N L
				 */
				
				// should NOT attempt to read MTD file for MatrixMarket format
				shouldReadMTD = false;
				
				// get metadata from MatrixMarket format file
				String[] headerLines = null;
				try {
					headerLines = IOUtilFunctions.readMatrixMarketHeader(inputFileName);
				}
				catch(DMLRuntimeException ex) {
					raiseValidateError(ex.getMessage(), conditional);
				}
				
				if (headerLines != null && headerLines.length >= 2){
					// process 1st line of MatrixMarket format to check for support types
					
					String firstLine = headerLines[0].trim();
					FileFormatPropertiesMM props = FileFormatPropertiesMM.parse(firstLine);
					
					// process 2nd line of MatrixMarket format -- must have size information
				
					String secondLine = headerLines[1];
					String[] sizeInfo = secondLine.trim().split("\\s+");
					if (sizeInfo.length != 3){
						raiseValidateError("Unsupported size line in MatrixMarket file: " +
							headerLines[1] + ". Only supported format in MatrixMarket file has size line: <NUM ROWS> <NUM COLS> <NUM NON-ZEROS>, where each value is an integer.", conditional);
					}
				
					long rowsCount = Long.parseLong(sizeInfo[0]);
					if (rowsCount < 0)
						raiseValidateError("MM file: invalid number of rows: "+rowsCount);
					else if( getVarParam(READROWPARAM) != null ) {
						long rowsCount2 = Long.parseLong(getVarParam(READROWPARAM).toString());
						if( rowsCount2 != rowsCount )
							raiseValidateError("MM file: invalid specified number of rows: "+rowsCount2+" vs "+rowsCount);
					}
					addVarParam(READROWPARAM, new IntIdentifier(rowsCount, this));

					long colsCount = Long.parseLong(sizeInfo[1]);
					if (colsCount < 0)
						raiseValidateError("MM file: invalid number of columns: "+colsCount);
					else if( getVarParam(READCOLPARAM) != null ) {
						long colsCount2 = Long.parseLong(getVarParam(READCOLPARAM).toString());
						if( colsCount2 != colsCount )
							raiseValidateError("MM file: invalid specified number of columns: "+colsCount2+" vs "+colsCount);
					}
					addVarParam(READCOLPARAM, new IntIdentifier(colsCount, this));
					configObj.setDimensions(rowsCount, colsCount);

					long nnzCount = Long.parseLong(sizeInfo[2]) * (props.isSymmetric() ? 2 : 1);
					if (nnzCount < 0)
						raiseValidateError("MM file: invalid number of non-zeros: "+nnzCount);
					else if( getVarParam(READNNZPARAM) != null ) {
						long nnzCount2 = Long.parseLong(getVarParam(READNNZPARAM).toString());
						if( nnzCount2 != nnzCount )
							raiseValidateError("MM file: invalid specified number of non-zeros: "+nnzCount2+" vs "+nnzCount);
					}
					addVarParam(READNNZPARAM, new IntIdentifier(nnzCount, this));
					configObj.setNnz(nnzCount);
				}
			}
			
			boolean isCSV = (formatTypeString != null && formatTypeString.equalsIgnoreCase(FileFormat.CSV.toString()));
			
			if (shouldReadMTD){
				configObj = new MetaDataAll(mtdFileName, conditional, false);
				if (configObj.mtdExists()){
					_varParams = configObj.parseMetaDataFileParameters(mtdFileName, conditional, _varParams);
					inferredFormatType = true;
				}
				else {
					if(!isCSV){
						LOG.warn("Metadata file: " + new Path(mtdFileName) + " not provided");
					}
				}
			}
			
			if (isCSV){
				// there should be no MTD file for delimited file format
				shouldReadMTD = true;
				
				// Handle valid ParamNames.
				if( !inferredFormatType ){
					for (String key : _varParams.keySet()){
						if (! READ_VALID_PARAM_NAMES.contains(key))
						{	
							String msg = "Only parameters allowed are: " + READ_VALID_PARAM_NAMES;
							raiseValidateError("Invalid parameter " + key + " in read statement: " +
								toString() + ". " + msg, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
					}
				}

				//handle all csv default parameters
				handleCSVDefaultParam(DELIM_DELIMITER, ValueType.STRING, conditional);
				handleCSVDefaultParam(DELIM_FILL_VALUE, ValueType.FP64, conditional);
				handleCSVDefaultParam(DELIM_HAS_HEADER_ROW, ValueType.BOOLEAN, conditional);
				handleCSVDefaultParam(DELIM_FILL, ValueType.BOOLEAN, conditional);
				handleCSVDefaultParam(DELIM_NA_STRINGS, ValueType.STRING, conditional);
			}

			boolean isLIBSVM = false;
			isLIBSVM = (formatTypeString != null && formatTypeString.equalsIgnoreCase(FileFormat.LIBSVM.toString()));
			if (isLIBSVM) {
				 // Handle libsvm file format
				shouldReadMTD = true;
				
				// only allow IO_FILENAME, READROWPARAM, READCOLPARAM   
				// as valid parameters
				if( !inferredFormatType ) {
					for (String key : _varParams.keySet()) {
						if (!(key.equals(IO_FILENAME) || key.equals(FORMAT_TYPE) 
								|| key.equals(READROWPARAM) || key.equals(READCOLPARAM)
								|| key.equals(READNNZPARAM) || key.equals(DATATYPEPARAM) 
								|| key.equals(VALUETYPEPARAM) || key.equals(DELIM_DELIMITER)
								|| key.equals(LIBSVM_INDEX_DELIM)))
						{	
							String msg = "Only parameters allowed are: " + IO_FILENAME + "," 
									+ READROWPARAM + "," + READCOLPARAM
									+ DELIM_DELIMITER + "," + LIBSVM_INDEX_DELIM;
							
							raiseValidateError("Invalid parameter " + key + " in read statement: " +
									toString() + ". " + msg, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
						}
					}
				}
				//handle all default parameters
				handleCSVDefaultParam(DELIM_DELIMITER, ValueType.STRING, conditional);
				handleCSVDefaultParam(LIBSVM_INDEX_DELIM, ValueType.STRING, conditional);
			}
			
			boolean isHDF5 = (formatTypeString != null && formatTypeString.equalsIgnoreCase(FileFormat.HDF5.toString()));

			boolean isCOG = (formatTypeString != null && formatTypeString.equalsIgnoreCase(FileFormat.COG.toString()));

			dataTypeString = (getVarParam(DATATYPEPARAM) == null) ? null : getVarParam(DATATYPEPARAM).toString();
			
			if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE) 
					|| dataTypeString.equalsIgnoreCase(Statement.FRAME_DATA_TYPE)) {
				
				boolean isMatrix = false;
				if ( dataTypeString == null || dataTypeString.equalsIgnoreCase(Statement.MATRIX_DATA_TYPE))
						isMatrix = true;
				
				// set data type
				getOutput().setDataType(isMatrix ? DataType.MATRIX : DataType.FRAME);

				// set number non-zeros
				Expression ennz = getVarParam("nnz");
				long nnz = -1;
				if( ennz != null ) {
					nnz = Long.valueOf(ennz.toString());
					getOutput().setNnz(nnz);
				}

				// Following dimension checks must be done when data type = MATRIX_DATA_TYPE 
				// initialize size of target data identifier to UNKNOWN
				getOutput().setDimensions(-1, -1);
				
				if (!isCSV && !isLIBSVM && !isHDF5 && !isCOG && ConfigurationManager.getCompilerConfig()
						.getBool(ConfigType.REJECT_READ_WRITE_UNKNOWNS) //skip check for csv/libsvm format / jmlc api
					&& (getVarParam(READROWPARAM) == null || getVarParam(READCOLPARAM) == null) ) {
						raiseValidateError("Missing or incomplete dimension information in read statement: "
								+ mtdFileName, conditional, LanguageErrorCodes.INVALID_PARAMETERS);
				}
				
				if (getVarParam(READROWPARAM) instanceof ConstIdentifier 
					&& getVarParam(READCOLPARAM) instanceof ConstIdentifier)
				{
					// these are strings that are long values
					Long dim1 = (getVarParam(READROWPARAM) == null) ? null : Long.valueOf( getVarParam(READROWPARAM).toString());
					Long dim2 = (getVarParam(READCOLPARAM) == null) ? null : Long.valueOf( getVarParam(READCOLPARAM).toString());
					if ( !isCSV && (dim1 < 0 || dim2 < 0) && ConfigurationManager
							.getCompilerConfig().getBool(ConfigType.REJECT_READ_WRITE_UNKNOWNS) ) {
						raiseValidateError("Invalid dimension information in read statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
					
					// set dim1 and dim2 values 
					if (dim1 != null && dim2 != null){
						getOutput().setDimensions(dim1, dim2);
					} else if (!isCSV && ((dim1 != null) || (dim2 != null))) {
						raiseValidateError("Partial dimension information in read statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
				}
				
				if(isLIBSVM) {
					Long dim2 = (getVarParam(READCOLPARAM) == null) ? null : Long.valueOf(getVarParam(READCOLPARAM).toString());
					if(dim2 < 0 && ConfigurationManager.getCompilerConfig()
							.getBool(ConfigType.REJECT_READ_WRITE_UNKNOWNS)) {
						raiseValidateError("Invalid dimension information in read statement", conditional, LanguageErrorCodes.INVALID_PARAMETERS);
					}
					getOutput().setDimensions(-1, dim2 + 1);
				}
				
				// initialize block dimensions to UNKNOWN 
				getOutput().setBlocksize(-1);
				
				String fmt =  (getVarParam(FORMAT_TYPE) == null ?
					FileFormat.defaultFormatString() : getVarParam(FORMAT_TYPE).toString());
				try {
					getOutput().setFileFormat(FileFormat.safeValueOf(fmt));
				}
				catch(Exception ex) {
					raiseValidateError("Invalid format '" + fmt+ "' in statement: " + toString(), conditional);
				}
				
				if (getVarParam(ROWBLOCKCOUNTPARAM) instanceof ConstIdentifier && getVarParam(COLUMNBLOCKCOUNTPARAM) instanceof ConstIdentifier)  {
					Integer rowBlockCount = (getVarParam(ROWBLOCKCOUNTPARAM) == null) ?
						null : Integer.valueOf(getVarParam(ROWBLOCKCOUNTPARAM).toString());
					getOutput().setBlocksize(rowBlockCount != null ? rowBlockCount : -1);
				}
				
				// block dimensions must be -1x-1 when format="text"
				// NOTE MB: disabled validate of default blocksize for inputs w/ format="binary"
				// because we automatically introduce reblocks if blocksizes don't match
				if ( (getOutput().getFileFormat().isTextFormat() || !isMatrix)  && getOutput().getBlocksize() != -1 ){
					raiseValidateError("Invalid block dimensions (" + getOutput().getBlocksize() + ") when format=" + getVarParam(FORMAT_TYPE) + " in \"" + this.toString() + "\".", conditional);
				}
			
			}
			else if ( dataTypeString.equalsIgnoreCase(Statement.SCALAR_DATA_TYPE)) {
				getOutput().setDataType(DataType.SCALAR);
				getOutput().setNnz(-1L);
			}
			else if ( dataTypeString.equalsIgnoreCase(DataType.LIST.name())) {
				getOutput().setDataType(DataType.LIST);
			}
			else{
				raiseValidateError("Unknown Data Type " + dataTypeString + ". Valid  values: " 
					+ Statement.SCALAR_DATA_TYPE +", " + Statement.MATRIX_DATA_TYPE+", " + Statement.FRAME_DATA_TYPE
					+", " + DataType.LIST.name().toLowerCase(), conditional, LanguageErrorCodes.INVALID_PARAMETERS);
			}
			
			// handle value type parameter
			if (getVarParam(VALUETYPEPARAM) != null && !(getVarParam(VALUETYPEPARAM) instanceof StringIdentifier)){
				raiseValidateError("for read method, parameter " + VALUETYPEPARAM + " can only be a string. " +
						"Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE, conditional);
			}
			// Identify the value type (used only for read method)
			String valueTypeString = getVarParam(VALUETYPEPARAM) == null ? null :  getVarParam(VALUETYPEPARAM).toString();
			if (valueTypeString != null) {
				if (valueTypeString.equalsIgnoreCase(Statement.DOUBLE_VALUE_TYPE))
					getOutput().setValueType(ValueType.FP64);
				else if (valueTypeString.equalsIgnoreCase(Statement.STRING_VALUE_TYPE))
					getOutput().setValueType(ValueType.STRING);
				else if (valueTypeString.equalsIgnoreCase(Statement.INT_VALUE_TYPE))
					getOutput().setValueType(ValueType.INT64);
				else if (valueTypeString.equalsIgnoreCase(Statement.BOOLEAN_VALUE_TYPE))
					getOutput().setValueType(ValueType.BOOLEAN);
				else if (valueTypeString.equalsIgnoreCase(ValueType.UNKNOWN.name()))
					getOutput().setValueType(ValueType.UNKNOWN);
				else {
					raiseValidateError("Unknown Value Type " + valueTypeString
						+ ". Valid values are: " + Statement.DOUBLE_VALUE_TYPE +", " + Statement.INT_VALUE_TYPE + ", " + Statement.BOOLEAN_VALUE_TYPE + ", " + Statement.STRING_VALUE_TYPE, conditional);
				}
			} else {
				getOutput().setValueType(ValueType.FP64);
			}

			break; 
			
		case WRITE:
			
			// for CSV format, if no delimiter specified THEN set default ","
			if (getVarParam(FORMAT_TYPE) == null || checkFormatType(FileFormat.CSV) ){
				if (getVarParam(DELIM_DELIMITER) == null) {
					addVarParam(DELIM_DELIMITER, new StringIdentifier(DEFAULT_DELIM_DELIMITER, this));
				}
				if (getVarParam(DELIM_HAS_HEADER_ROW) == null) {
					addVarParam(DELIM_HAS_HEADER_ROW, new BooleanIdentifier(DEFAULT_DELIM_HAS_HEADER_ROW, this));
				}
				if (getVarParam(DELIM_SPARSE) == null) {
					addVarParam(DELIM_SPARSE, new BooleanIdentifier(DEFAULT_DELIM_SPARSE, this));
				}
			}
			
			// for LIBSVM format, add the default separators if not specified
			if (getVarParam(FORMAT_TYPE) == null || checkFormatType(FileFormat.LIBSVM)) {
				if(getVarParam(DELIM_DELIMITER) == null) {
					addVarParam(DELIM_DELIMITER, new StringIdentifier(DEFAULT_DELIM_DELIMITER, this));
				}
				if(getVarParam(LIBSVM_INDEX_DELIM) == null) {
					addVarParam(LIBSVM_INDEX_DELIM, new StringIdentifier(DEFAULT_LIBSVM_INDEX_DELIM, this));
				}
				if(getVarParam(DELIM_SPARSE) == null) {
					addVarParam(DELIM_SPARSE, new BooleanIdentifier(DEFAULT_DELIM_SPARSE, this));
				}
			}
			
			//validate read filename
			if (getVarParam(FORMAT_TYPE) == null || FileFormat.isTextFormat(getVarParam(FORMAT_TYPE).toString()))
				getOutput().setBlocksize(-1);
			else if (checkFormatType(FileFormat.BINARY, FileFormat.COMPRESSED, FileFormat.UNKNOWN)) {
				if( getVarParam(ROWBLOCKCOUNTPARAM)!=null )
					getOutput().setBlocksize(Integer.parseInt(getVarParam(ROWBLOCKCOUNTPARAM).toString()));
				else
					getOutput().setBlocksize(ConfigurationManager.getBlocksize());
			}
			else if( getVarParam(FORMAT_TYPE) instanceof StringIdentifier ) //literal format
				raiseValidateError("Invalid format " + getVarParam(FORMAT_TYPE)
					+ " in statement: " + toString(), conditional);
			break;

			case RAND:
			
			Expression dataParam = getVarParam(RAND_DATA);
			
			if( dataParam != null )
			{
				// handle input variable (matrix/scalar)
				if( dataParam instanceof DataIdentifier )
				{
					addVarParam(RAND_MIN, dataParam);
					addVarParam(RAND_MAX, dataParam);
				}
				// handle integer constant
				else if (dataParam instanceof IntIdentifier) {
					addVarParam(RAND_MIN, dataParam);
					addVarParam(RAND_MAX, dataParam);
				}
				// handle double constant
				else if (dataParam instanceof DoubleIdentifier) {
					double roundedValue = ((DoubleIdentifier)dataParam).getValue();
					Expression minExpr = new DoubleIdentifier(roundedValue, this);
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);
				}
				// handle string constant (string init)
				else if (dataParam instanceof StringIdentifier) {
					String data = ((StringIdentifier)dataParam).getValue();
					Expression minExpr = new StringIdentifier(data, this);
					addVarParam(RAND_MIN, minExpr);
					addVarParam(RAND_MAX, minExpr);
					_strInit = true;
				}
				else {
					// handle general expression
					dataParam.validateExpression(ids, currConstVars, conditional);
					addVarParam(RAND_MIN, dataParam);
					addVarParam(RAND_MAX, dataParam);
				}
				
				removeVarParam(RAND_DATA);
				removeVarParam(RAND_BY_ROW);
				this.setRandDefault();
			}
			
			//check valid parameters
			validateParams(conditional, RAND_VALID_PARAM_NAMES, "Legal parameters for Rand statement are "
					+ "(capitalization-sensitive): " 	+ RAND_ROWS + ", " + RAND_COLS + ", " + RAND_DIMS + ", "
					+ RAND_MIN + ", " + RAND_MAX + ", " + RAND_SPARSITY + ", " + RAND_SEED     + ", "
					+ RAND_PDF  + ", " + RAND_LAMBDA);

			//parameters w/ support for variable inputs
			if (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier){
				raiseValidateError("for Rand statement " + RAND_ROWS + " has incorrect value type", conditional);
			}
			
			if (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier){
				raiseValidateError("for Rand statement " + RAND_COLS + " has incorrect value type", conditional);
			}

			if (getVarParam(RAND_DIMS) instanceof IntIdentifier || getVarParam(RAND_DIMS) instanceof DoubleIdentifier
					|| getVarParam(RAND_DIMS) instanceof BooleanIdentifier){
				raiseValidateError("for Rand statement " + RAND_DIMS + " has incorrect value type", conditional);
			}

			if (getVarParam(RAND_SEED) instanceof StringIdentifier || getVarParam(RAND_SEED) instanceof BooleanIdentifier) {
				raiseValidateError("for Rand statement " + RAND_SEED + " has incorrect value type", conditional);
			}

			boolean isTensorOperation = getVarParam(RAND_DIMS) != null;

			if ((getVarParam(RAND_MAX) instanceof StringIdentifier && !_strInit) ||
					(getVarParam(RAND_MAX) instanceof BooleanIdentifier && !isTensorOperation)) {
				raiseValidateError("for Rand statement " + RAND_MAX + " has incorrect value type", conditional);
			}

			if ((getVarParam(RAND_MIN) instanceof StringIdentifier && !_strInit) ||
					getVarParam(RAND_MIN) instanceof BooleanIdentifier && !isTensorOperation)
				raiseValidateError("for Rand statement " + RAND_MIN + " has incorrect value type", conditional);

			// Since sparsity can be arbitrary expression (SYSTEMML-515), no validation check for DoubleIdentifier/IntIdentifier required.
			
			if (!(getVarParam(RAND_PDF) instanceof StringIdentifier)) {
				raiseValidateError("for Rand statement " + RAND_PDF + " has incorrect value type", conditional);
			}
	
			Expression lambda = getVarParam(RAND_LAMBDA);
			if (!( (lambda instanceof DataIdentifier
					|| lambda instanceof ConstIdentifier)
				&& (lambda.getOutput().getValueType() == ValueType.FP64
					|| lambda.getOutput().getValueType() == ValueType.INT64) )) {
				raiseValidateError("for Rand statement " + RAND_LAMBDA + " has incorrect data type", conditional);
			}
				
			long rowsLong = -1L, colsLong = -1L;
			
			Expression rowsExpr = getVarParam(RAND_ROWS);
			Expression colsExpr = getVarParam(RAND_COLS);
			if (!isTensorOperation) {
				///////////////////////////////////////////////////////////////////
				// HANDLE ROWS
				///////////////////////////////////////////////////////////////////
				if( rowsExpr instanceof IntIdentifier ) {
					if( ((IntIdentifier) rowsExpr).getValue() < 0 ) {
						raiseValidateError("In rand statement, can only assign rows a long " +
								"(integer) value >= 0 -- attempted to assign value: " + ((IntIdentifier) rowsExpr).getValue(), conditional);
					}
					rowsLong = ((IntIdentifier) rowsExpr).getValue();
				}
				else if( rowsExpr instanceof DoubleIdentifier ) {
					if( ((DoubleIdentifier) rowsExpr).getValue() < 0 ) {
						raiseValidateError("In rand statement, can only assign rows a long " +
								"(integer) value >= 0 -- attempted to assign value: " + rowsExpr.toString(), conditional);
					}
					rowsLong = UtilFunctions.toLong(Math.floor(((DoubleIdentifier) rowsExpr).getValue()));
				}
				else if( rowsExpr instanceof DataIdentifier && !(rowsExpr instanceof IndexedIdentifier) ) {
					
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier) rowsExpr).getName();
					if( currConstVars.containsKey(identifierName) ) {
						
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if( constValue instanceof IntIdentifier ) {
							// check rows is >= 1 --- throw exception
							if( ((IntIdentifier) constValue).getValue() < 0 ) {
								raiseValidateError("In rand statement, can only assign rows a long " +
										"(integer) value >= 0 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier
							long roundedValue = ((IntIdentifier) constValue).getValue();
							rowsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue;
						}
						// handle double constant
						else if( constValue instanceof DoubleIdentifier ) {
							if( ((DoubleIdentifier) constValue).getValue() < 0 ) {
								raiseValidateError("In rand statement, can only assign rows a long " +
										"(double) value >= 0 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier) constValue).getValue())).longValue();
							rowsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue;
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In rand statement, can only assign rows a long " +
									"(integer) value >= 0 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						rowsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars, conditional);
				}
				
				///////////////////////////////////////////////////////////////////
				// HANDLE COLUMNS
				///////////////////////////////////////////////////////////////////
				if( colsExpr instanceof IntIdentifier ) {
					if( ((IntIdentifier) colsExpr).getValue() < 0 ) {
						raiseValidateError("In rand statement, can only assign cols a long " +
								"(integer) value >= 0 -- attempted to assign value: " + colsExpr.toString(), conditional);
					}
					colsLong = ((IntIdentifier) colsExpr).getValue();
				}
				else if( colsExpr instanceof DoubleIdentifier ) {
					if( ((DoubleIdentifier) colsExpr).getValue() < 0 ) {
						raiseValidateError("In rand statement, can only assign cols a long " +
								"(integer) value >= 0 -- attempted to assign value: " + colsExpr.toString(), conditional);
					}
					colsLong = Double.valueOf((Math.floor(((DoubleIdentifier) colsExpr).getValue()))).longValue();
				}
				else if( colsExpr instanceof DataIdentifier && !(colsExpr instanceof IndexedIdentifier) ) {
					
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier) colsExpr).getName();
					if( currConstVars.containsKey(identifierName) ) {
						
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if( constValue instanceof IntIdentifier ) {
							if( ((IntIdentifier) constValue).getValue() < 0 ) {
								raiseValidateError("In rand statement, can only assign cols a long " +
										"(integer) value >= 0 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier
							long roundedValue = ((IntIdentifier) constValue).getValue();
							colsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue;
						}
						// handle double constant
						else if( constValue instanceof DoubleIdentifier ) {
							if( ((DoubleIdentifier) constValue).getValue() < 0 ) {
								raiseValidateError("In rand statement, can only assign cols a long " +
										"(double) value >= 0 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier) constValue).getValue())).longValue();
							colsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue;
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In rand statement, can only assign cols a long " +
									"(integer) value >= 0 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						colsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars, conditional);
				}
			}
			///////////////////////////////////////////////////////////////////
			// HANDLE MIN
			///////////////////////////////////////////////////////////////////
			Expression minExpr = getVarParam(RAND_MIN);
			
			// perform constant propogation
			if (minExpr instanceof DataIdentifier && !(minExpr instanceof IndexedIdentifier)) {
				
				// check if the DataIdentifier variable is a ConstIdentifier
				String identifierName = ((DataIdentifier)minExpr).getName();
				if (currConstVars.containsKey(identifierName)){
					
					// handle int constant
					ConstIdentifier constValue = currConstVars.get(identifierName);
					if (constValue instanceof IntIdentifier){
						
						// update min expr with new IntIdentifier
						long roundedValue = ((IntIdentifier)constValue).getValue();
						minExpr = new DoubleIdentifier(roundedValue, this);
						addVarParam(RAND_MIN, minExpr);
					}
					// handle double constant
					else if (constValue instanceof DoubleIdentifier){
		
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						minExpr = new DoubleIdentifier(roundedValue, this);
						addVarParam(RAND_MIN, minExpr);
						
					}
					else {
						// exception -- rows must be integer or double constant
						raiseValidateError("In rand statement, can only assign min a numerical " +
								"value -- attempted to assign: " + constValue.toString(), conditional);
					}
				}
				else {
					// handle general expression
					minExpr.validateExpression(ids, currConstVars, conditional);
				}
			}
			else {
				// handle general expression
				minExpr.validateExpression(ids, currConstVars, conditional);
			}
			
			
			///////////////////////////////////////////////////////////////////
			// HANDLE MAX
			///////////////////////////////////////////////////////////////////
			Expression maxExpr = getVarParam(RAND_MAX);
			
			// perform constant propogation
			if (maxExpr instanceof DataIdentifier && !(maxExpr instanceof IndexedIdentifier)) {
				
				// check if the DataIdentifier variable is a ConstIdentifier
				String identifierName = ((DataIdentifier)maxExpr).getName();
				if (currConstVars.containsKey(identifierName)) {
					// handle int constant
					ConstIdentifier constValue = currConstVars.get(identifierName);
					if (constValue instanceof IntIdentifier) {
						// update min expr with new IntIdentifier
						long roundedValue = ((IntIdentifier)constValue).getValue();
						maxExpr = new DoubleIdentifier(roundedValue, this);
						addVarParam(RAND_MAX, maxExpr);
					}
					// handle double constant
					else if (constValue instanceof DoubleIdentifier) {
						// update col expr with new IntIdentifier (rounded down)
						double roundedValue = ((DoubleIdentifier)constValue).getValue();
						maxExpr = new DoubleIdentifier(roundedValue, this);
						addVarParam(RAND_MAX, maxExpr);
					}
					else {
						// exception -- rows must be integer or double constant
						raiseValidateError("In rand statement, can only assign max a numerical " +
								"value -- attempted to assign: " + constValue.toString(), conditional);
					}
				}
				else {
					// handle general expression
					maxExpr.validateExpression(ids, currConstVars, conditional);
				}
			}
			else {
				// handle general expression
				maxExpr.validateExpression(ids, currConstVars, conditional);
			}
		
			getOutput().setFileFormat(FileFormat.BINARY);
			if (isTensorOperation) {
				getOutput().setDataType(DataType.TENSOR);
				getOutput().setValueType(getVarParam(RAND_MIN).getOutput().getValueType());
				// TODO set correct dimensions
				getOutput().setDimensions(-1, -1);
			} else {
				getOutput().setDataType(DataType.MATRIX);
				getOutput().setValueType(ValueType.FP64);
				getOutput().setDimensions(rowsLong, colsLong);
			}
			
			if (getOutput() instanceof IndexedIdentifier){
				// process the "target" being indexed
				DataIdentifier targetAsSeen = ids.get(((DataIdentifier)getOutput()).getName());
				if (targetAsSeen == null){
					raiseValidateError("cannot assign value to indexed identifier " + ((DataIdentifier)getOutput()).getName() + " without first initializing " + ((DataIdentifier)getOutput()).getName(), conditional);
				}
				//_output.setProperties(targetAsSeen);
				((IndexedIdentifier) getOutput()).setOriginalDimensions(targetAsSeen.getDim1(), targetAsSeen.getDim2());
				//((IndexedIdentifier) getOutput()).setOriginalDimensions(getOutput().getDim1(), getOutput().getDim2());
			}
			if (getOutput() instanceof IndexedIdentifier){
				LOG.warn(this.printWarningLocation() + "Output for Rand Statement may have incorrect size information");
			}
			
			break;
			
		case MATRIX: 
			
			//handle default and input arguments
			setMatrixDefault();
			validateParams(conditional, RESHAPE_VALID_PARAM_NAMES,
					"Legal parameters for matrix statement are (case-sensitive): "
						+ RAND_DATA + ", " + RAND_ROWS	+ ", " + RAND_COLS + ", " + RAND_BY_ROW);

			//validate correct value types
			if (getVarParam(RAND_DATA) != null && (getVarParam(RAND_DATA) instanceof BooleanIdentifier)){
				raiseValidateError("for matrix statement " + RAND_DATA + " has incorrect value type", conditional);
			}
			if (getVarParam(RAND_ROWS) != null && (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier)){
				raiseValidateError("for matrix statement " + RAND_ROWS + " has incorrect value type", conditional);
			}				
			if (getVarParam(RAND_COLS) != null && (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier)){
				raiseValidateError("for matrix statement " + RAND_COLS + " has incorrect value type", conditional);
			}				
			if ( !(getVarParam(RAND_BY_ROW) instanceof BooleanIdentifier)) {
				raiseValidateError("for matrix statement " + RAND_BY_ROW + " has incorrect value type", conditional);
			}
			
			//validate general data expression
			getVarParam(RAND_DATA).validateExpression(ids, currConstVars, conditional);
			
			rowsLong = -1L; 
			colsLong = -1L;

			///////////////////////////////////////////////////////////////////
			// HANDLE ROWS
			///////////////////////////////////////////////////////////////////
			rowsExpr = getVarParam(RAND_ROWS);
			if (rowsExpr != null){
				if (rowsExpr instanceof IntIdentifier) {
					if  (((IntIdentifier)rowsExpr).getValue() >= 1 ) {
						rowsLong = ((IntIdentifier)rowsExpr).getValue();
					}
					else {
						raiseValidateError("In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue(), conditional);
					}
				}
				else if (rowsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 ) {
						rowsLong = Double.valueOf((Math.floor(((DoubleIdentifier)rowsExpr).getValue()))).longValue();
					}
					else {
						raiseValidateError("In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString(), conditional);
					}		
				}
				else if (rowsExpr instanceof DataIdentifier && !(rowsExpr instanceof IndexedIdentifier)) {
					
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier)rowsExpr).getName();
					if (currConstVars.containsKey(identifierName)){
						
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if (constValue instanceof IntIdentifier){
							
							// check rows is >= 1 --- throw exception
							if (((IntIdentifier)constValue).getValue() < 1){
								raiseValidateError("In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier 
							long roundedValue = ((IntIdentifier)constValue).getValue();
							rowsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue; 
						}
						// handle double constant 
						else if (constValue instanceof DoubleIdentifier){
							
							if (((DoubleIdentifier)constValue).getValue() < 1.0){
								raiseValidateError("In matrix statement, can only assign rows a long " +
										"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							rowsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue; 
							
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In matrix statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						rowsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}	
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars, conditional);
				}
			}
	
			///////////////////////////////////////////////////////////////////
			// HANDLE COLUMNS
			///////////////////////////////////////////////////////////////////
			
			colsExpr = getVarParam(RAND_COLS);
			if (colsExpr != null){
				if (colsExpr instanceof IntIdentifier) {
					if  (((IntIdentifier)colsExpr).getValue() >= 1 ) {
						colsLong = ((IntIdentifier)colsExpr).getValue();
					}
					else {
						raiseValidateError("In matrix statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
					}
				}
				else if (colsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)colsExpr).getValue() >= 1 ) {
						colsLong = Double.valueOf((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
					}
					else {
						raiseValidateError("In matrix statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
					}
				}
				else if (colsExpr instanceof DataIdentifier && !(colsExpr instanceof IndexedIdentifier)) {
					
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier)colsExpr).getName();
					if (currConstVars.containsKey(identifierName)){
						
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if (constValue instanceof IntIdentifier){
							
							// check cols is >= 1 --- throw exception
							if (((IntIdentifier)constValue).getValue() < 1){
								raiseValidateError("In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " 
										+ constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier 
							long roundedValue = ((IntIdentifier)constValue).getValue();
							colsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue; 
						}
						// handle double constant 
						else if (constValue instanceof DoubleIdentifier){
							if (((DoubleIdentifier)constValue).getValue() < 1){
								raiseValidateError("In matrix statement, can only assign cols a long " +
										"(integer) value >= 1 -- attempted to assign value: " 
										+ constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							colsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue;
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In matrix statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						colsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars, conditional);
				}
			}
			getOutput().setFileFormat(FileFormat.BINARY);
			getOutput().setDataType(DataType.MATRIX);
			getOutput().setValueType(ValueType.FP64);
			getOutput().setDimensions(rowsLong, colsLong);
			
			if (getOutput() instanceof IndexedIdentifier){
				((IndexedIdentifier) getOutput()).setOriginalDimensions(getOutput().getDim1(), getOutput().getDim2());
				LOG.warn(this.printWarningLocation() + "Output for matrix Statement may have incorrect size information");
			}
			
			break;

		case FRAME:
			//handle default and input arguments
			setFrameDefault();
			validateParams(conditional, FRAME_VALID_PARAM_NAMES,
				"Legal parameters for frame statement are (case-sensitive): "
					+ RAND_DATA + ", " + RAND_ROWS	+ ", " + RAND_COLS + ", " + SCHEMAPARAM);

			//validate correct value types
			if (getVarParam(RAND_ROWS) != null && (getVarParam(RAND_ROWS) instanceof StringIdentifier || getVarParam(RAND_ROWS) instanceof BooleanIdentifier)){
				raiseValidateError("for frame statement " + RAND_ROWS + " has incorrect value type", conditional);
			}
			if (getVarParam(RAND_COLS) != null && (getVarParam(RAND_COLS) instanceof StringIdentifier || getVarParam(RAND_COLS) instanceof BooleanIdentifier)){
				raiseValidateError("for frame statement " + RAND_COLS + " has incorrect value type", conditional);
			}

			//validate general data expression
			getVarParam(RAND_DATA).validateExpression(ids, currConstVars, conditional);

			rowsLong = -1L;
			colsLong = -1L;

			///////////////////////////////////////////////////////////////////
			// HANDLE ROWS
			///////////////////////////////////////////////////////////////////
			rowsExpr = getVarParam(RAND_ROWS);
			if (rowsExpr != null){
				if (rowsExpr instanceof IntIdentifier) {
					if  (((IntIdentifier)rowsExpr).getValue() >= 1 )
						rowsLong = ((IntIdentifier)rowsExpr).getValue();
					else
						raiseValidateError("In frame statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + ((IntIdentifier)rowsExpr).getValue(), conditional);
				}
				else if (rowsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)rowsExpr).getValue() >= 1 )
						rowsLong = Double.valueOf((Math.floor(((DoubleIdentifier)rowsExpr).getValue()))).longValue();
					else
						raiseValidateError("In frame statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + rowsExpr.toString(), conditional);
				}
				else if (rowsExpr instanceof DataIdentifier && !(rowsExpr instanceof IndexedIdentifier)) {
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier)rowsExpr).getName();
					if (currConstVars.containsKey(identifierName)){
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if (constValue instanceof IntIdentifier){
							// check rows is >= 1 --- throw exception
							if (((IntIdentifier)constValue).getValue() < 1){
								raiseValidateError("In frame statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier
							long roundedValue = ((IntIdentifier)constValue).getValue();
							rowsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue;
						}
						// handle double constant
						else if (constValue instanceof DoubleIdentifier){
							if (((DoubleIdentifier)constValue).getValue() < 1.0){
								raiseValidateError("In frame statement, can only assign rows a long " +
									"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
							}
							// update row expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							rowsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_ROWS, rowsExpr);
							rowsLong = roundedValue;
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In frame statement, can only assign rows a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						rowsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}
				else {
					// handle general expression
					rowsExpr.validateExpression(ids, currConstVars, conditional);
				}
			}

			///////////////////////////////////////////////////////////////////
			// HANDLE COLUMNS
			///////////////////////////////////////////////////////////////////

			colsExpr = getVarParam(RAND_COLS);
			if (colsExpr != null){
				if (colsExpr instanceof IntIdentifier) {
					if  (((IntIdentifier)colsExpr).getValue() >= 1 )
						colsLong = ((IntIdentifier)colsExpr).getValue();
					else
						raiseValidateError("In frame statement, can only assign cols a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
				}
				else if (colsExpr instanceof DoubleIdentifier) {
					if  (((DoubleIdentifier)colsExpr).getValue() >= 1 )
						colsLong = Double.valueOf((Math.floor(((DoubleIdentifier)colsExpr).getValue()))).longValue();
					else
						raiseValidateError("In frame statement, can only assign rows a long " +
							"(integer) value >= 1 -- attempted to assign value: " + colsExpr.toString(), conditional);
				}
				else if (colsExpr instanceof DataIdentifier && !(colsExpr instanceof IndexedIdentifier)) {
					// check if the DataIdentifier variable is a ConstIdentifier
					String identifierName = ((DataIdentifier)colsExpr).getName();
					if (currConstVars.containsKey(identifierName)){
						// handle int constant
						ConstIdentifier constValue = currConstVars.get(identifierName);
						if (constValue instanceof IntIdentifier){
							// check cols is >= 1 --- throw exception
							if (((IntIdentifier)constValue).getValue() < 1){
								raiseValidateError("In frame statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: "
									+ constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier
							long roundedValue = ((IntIdentifier)constValue).getValue();
							colsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue;
						}
						// handle double constant
						else if (constValue instanceof DoubleIdentifier){
							if (((DoubleIdentifier)constValue).getValue() < 1){
								raiseValidateError("In frame statement, can only assign cols a long " +
									"(integer) value >= 1 -- attempted to assign value: "
									+ constValue.toString(), conditional);
							}
							// update col expr with new IntIdentifier (rounded down)
							long roundedValue = Double.valueOf(Math.floor(((DoubleIdentifier)constValue).getValue())).longValue();
							colsExpr = new IntIdentifier(roundedValue, this);
							addVarParam(RAND_COLS, colsExpr);
							colsLong = roundedValue;
						}
						else {
							// exception -- rows must be integer or double constant
							raiseValidateError("In frame statement, can only assign cols a long " +
								"(integer) value >= 1 -- attempted to assign value: " + constValue.toString(), conditional);
						}
					}
					else {
						// handle general expression
						colsExpr.validateExpression(ids, currConstVars, conditional);
					}
				}
				else {
					// handle general expression
					colsExpr.validateExpression(ids, currConstVars, conditional);
				}
			}
			getOutput().setFileFormat(FileFormat.BINARY);
			getOutput().setDataType(DataType.FRAME);
			getOutput().setValueType(ValueType.UNKNOWN);
			getOutput().setDimensions(rowsLong, colsLong);

			if (getOutput() instanceof IndexedIdentifier){
				((IndexedIdentifier) getOutput()).setOriginalDimensions(getOutput().getDim1(), getOutput().getDim2());
				LOG.warn(this.printWarningLocation() + "Output for frame Statement may have incorrect size information");
			}
			break;
			
		case TENSOR:
			//handle default and input arguments
			setTensorDefault();
			validateParams(conditional, RESHAPE_VALID_PARAM_NAMES,
					"Legal parameters for tensor statement are (case-sensitive): "
						+ RAND_DATA + ", " + RAND_DIMS	+ ", " + RAND_BY_ROW);

			//validate correct value types
			/*if (getVarParam(RAND_DATA) != null && (getVarParam(RAND_DATA) instanceof BooleanIdentifier)){
				raiseValidateError("for tensor statement " + RAND_DATA + " has incorrect value type", conditional);
			}*/
			if (getVarParam(RAND_DIMS) != null && (getVarParam(RAND_DIMS) instanceof BooleanIdentifier)){
				raiseValidateError("for tensor statement " + RAND_DIMS + " has incorrect value type", conditional);
			}
			if ( !(getVarParam(RAND_BY_ROW) instanceof BooleanIdentifier)) {
				raiseValidateError("for tensor statement " + RAND_BY_ROW + " has incorrect value type", conditional);
			}

			//validate general data expression
			getVarParam(RAND_DATA).validateExpression(ids, currConstVars, conditional);
			getVarParam(RAND_DIMS).validateExpression(ids, currConstVars, conditional);

			getOutput().setFileFormat(FileFormat.BINARY);
			getOutput().setDataType(DataType.TENSOR);
			getOutput().setValueType(getVarParam(RAND_DATA).getOutput().getValueType());
			// TODO get size
			getOutput().setDimensions(-1, -1);

			if (getOutput() instanceof IndexedIdentifier){
				((IndexedIdentifier) getOutput()).setOriginalDimensions(getOutput().getDim1(), getOutput().getDim2());
			}
			//getOutput().computeDataType();

			if (getOutput() instanceof IndexedIdentifier){
				LOG.warn(this.printWarningLocation() + "Output for tensor Statement may have incorrect size information");
			}
			break;

		case SQL:
			//handle default and input arguments
			setSqlDefault();
			validateParams(conditional, SQL_VALID_PARAM_NAMES,
					"Legal parameters for tensor statement are (case-sensitive): " + SQL_CONN + ", " +
							SQL_USER + ", " + SQL_PASS + ", " + SQL_QUERY);
			
			//validate correct value types
			Expression exp = getVarParam(SQL_CONN);
			if( !(exp instanceof StringIdentifier) && exp instanceof Identifier ) {
				raiseValidateError("for tensor statement " + SQL_CONN + " has incorrect value type", conditional);
			}
			exp = getVarParam(SQL_USER);
			if( !(exp instanceof StringIdentifier) && exp instanceof Identifier ) {
				raiseValidateError("for tensor statement " + SQL_USER + " has incorrect value type", conditional);
			}
			exp = getVarParam(SQL_PASS);
			if( !(exp instanceof StringIdentifier) && exp instanceof Identifier ) {
				raiseValidateError("for tensor statement " + SQL_PASS + " has incorrect value type", conditional);
			}
			exp = getVarParam(SQL_QUERY);
			if( !(exp instanceof StringIdentifier) && exp instanceof Identifier ) {
				raiseValidateError("for tensor statement " + SQL_QUERY + " has incorrect value type", conditional);
			}
			
			//validate general data expression
			getVarParam(SQL_CONN).validateExpression(ids, currConstVars, conditional);
			getVarParam(SQL_USER).validateExpression(ids, currConstVars, conditional);
			getVarParam(SQL_PASS).validateExpression(ids, currConstVars, conditional);
			getVarParam(SQL_QUERY).validateExpression(ids, currConstVars, conditional);
			
			getOutput().setFileFormat(FileFormat.BINARY);
			getOutput().setDataType(DataType.TENSOR);
			getOutput().setValueType(ValueType.UNKNOWN);
			getOutput().setDimensions(-1, -1);
			
			if (getOutput() instanceof IndexedIdentifier){
				LOG.warn(this.printWarningLocation() + "Output for sql statement may have incorrect size information");
			}
			break;
			
		case FEDERATED:
			validateParams(conditional, FEDERATED_VALID_PARAM_NAMES,
				"Legal parameters for federated statement are (case-sensitive): "
				+ FED_TYPE + ", " + FED_ADDRESSES + ", " + FED_RANGES);
			exp = getVarParam(FED_ADDRESSES);
			if( !(exp instanceof DataIdentifier) ) {
				raiseValidateError("for federated statement " + FED_ADDRESSES + " has incorrect value type", conditional);
			}
			getVarParam(FED_ADDRESSES).validateExpression(ids, currConstVars, conditional);
			exp = getVarParam(FED_RANGES);
			if( !(exp instanceof DataIdentifier) ) {
				raiseValidateError("for federated statement " + FED_RANGES + " has incorrect value type", conditional);
			}
			getVarParam(FED_RANGES).validateExpression(ids, currConstVars, conditional);
			exp = getVarParam(FED_TYPE);
			if( !(exp instanceof StringIdentifier) ) {
				raiseValidateError("for federated statement " + FED_TYPE + " has incorrect value type", conditional);
			}
			getVarParam(FED_TYPE).validateExpression(ids, currConstVars, conditional);
			
			getOutput().setFileFormat(FileFormat.BINARY);
			StringIdentifier fedType = (StringIdentifier) exp;
			if(fedType.getValue().equalsIgnoreCase(FED_MATRIX_IDENTIFIER)) {
				getOutput().setDataType(DataType.MATRIX);
				// TODO value type for federated object
				getOutput().setValueType(ValueType.FP64);
			}
			else if(fedType.getValue().equalsIgnoreCase(FED_FRAME_IDENTIFIER)) {
				getOutput().setDataType(DataType.FRAME);
			}

			if(_varParams.size() == 4) {
				exp = getVarParam(FED_LOCAL_OBJECT);
				if( !(exp instanceof DataIdentifier) ) {
					raiseValidateError("for federated statement " + FED_LOCAL_OBJECT + " has incorrect value type", conditional);
				}
				getVarParam(FED_LOCAL_OBJECT).validateExpression(ids, currConstVars, conditional);
			}
			getOutput().setDimensions(-1, -1);

			break;
			
		default:
			raiseValidateError("Unsupported Data expression "+ this.getOpCode(), false, LanguageErrorCodes.INVALID_PARAMETERS); //always unconditional
		}
	}
	
	private void handleCSVDefaultParam(String param, ValueType vt, boolean conditional) {
		if (getVarParam(param) == null) {
			Expression defExpr = null;
			switch(vt) {
				case BOOLEAN:
					defExpr = new BooleanIdentifier((boolean)csvDefaults.get(param), this);
					break;
				case FP64:
					defExpr = new DoubleIdentifier((double)csvDefaults.get(param), this);
					break;
				default:
					defExpr = new StringIdentifier((String)csvDefaults.get(param), this);
			}
			addVarParam(param, defExpr);
		}
		else {
			if ( (getVarParam(param) instanceof ConstIdentifier)
				&& (! checkValueType(getVarParam(param), vt)))
			{
				raiseValidateError("For delimited file '" + getVarParam(param) 
					+ "' must be a '"+vt.toExternalString()+"' value ", conditional);
			}
		}
	}
	
	private boolean checkFormatType(FileFormat... fmts) {
		String fmtStr = getVarParam(FORMAT_TYPE).toString();
		return Arrays.stream(fmts)
			.anyMatch(fmt -> fmtStr.equalsIgnoreCase(fmt.toString()));
	}
	
	private boolean checkValueType(Expression expr, ValueType vt) {
		return (vt == ValueType.STRING && expr instanceof StringIdentifier)
			|| (vt == ValueType.FP64 && (expr instanceof DoubleIdentifier || expr instanceof IntIdentifier))
			|| (vt == ValueType.BOOLEAN && expr instanceof BooleanIdentifier);
	}

	private void validateParams(boolean conditional, Set<String> validParamNames, String legalMessage) {
		for( String key : _varParams.keySet() )
		{
			boolean found = validParamNames.contains(key);
			if( !found ) {
				raiseValidateError("unexpected parameter \"" + key + "\". "
						+ legalMessage, conditional);
			}
		}
	}

	private void performConstantPropagationRand( HashMap<String, ConstIdentifier> currConstVars )
	{
		//here, we propagate constants for all rand parameters that are required during validate.
		String[] paramNamesForEval = new String[]{RAND_DATA, RAND_SPARSITY, RAND_MIN, RAND_MAX};
		
		//replace data identifiers with const identifiers
		performConstantPropagation(currConstVars, paramNamesForEval);
	}

	private void performConstantPropagationReadWrite( HashMap<String, ConstIdentifier> currConstVars )
	{
		//here, we propagate constants for all read/write parameters that are required during validate.
		String[] paramNamesForEval = new String[]{FORMAT_TYPE, IO_FILENAME, READROWPARAM, READCOLPARAM, READNNZPARAM};
		
		//replace data identifiers with const identifiers
		performConstantPropagation(currConstVars, paramNamesForEval);
	}

	private void performConstantPropagation( HashMap<String, ConstIdentifier> currConstVars, String[] paramNames )
	{
		for( String paramName : paramNames )
		{
			Expression paramExp = getVarParam(paramName);
			if (   paramExp != null && paramExp instanceof DataIdentifier && !(paramExp instanceof IndexedIdentifier) 
				&& currConstVars.containsKey(((DataIdentifier) paramExp).getName()))
			{
				addVarParam(paramName, currConstVars.get(((DataIdentifier)paramExp).getName()));
			}
		}
	}
	
	
	private String fileNameCat(BinaryExpression expr, HashMap<String, ConstIdentifier> currConstVars, String filename, boolean conditional)
	{
		// Processing the left node first
		if (expr.getLeft() instanceof BinaryExpression 
			&& ((BinaryExpression)expr.getLeft()).getOpCode() == BinaryOp.PLUS){
			filename = fileNameCat((BinaryExpression)expr.getLeft(), currConstVars, filename, conditional)+ filename;
		}
		else if (expr.getLeft() instanceof ConstIdentifier){
			filename = ((ConstIdentifier)expr.getLeft()).toString()+ filename;
		}
		else if (expr.getLeft() instanceof DataIdentifier 
			&& ((DataIdentifier)expr.getLeft()).getDataType() == DataType.SCALAR){ 
			String name = ((DataIdentifier)expr.getLeft()).getName();
			filename = ((StringIdentifier)currConstVars.get(name)).getValue() + filename;
		}
		else {
			raiseValidateError("Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.", conditional);
		}
		// Now process the right node
		if (expr.getRight() instanceof BinaryExpression 
			&& ((BinaryExpression)expr.getRight()).getOpCode() == BinaryOp.PLUS){
			filename = filename + fileNameCat((BinaryExpression)expr.getRight(), currConstVars, filename, conditional);
		}
		// DRB: CHANGE
		else if (expr.getRight() instanceof ConstIdentifier){
			filename = filename + ((ConstIdentifier)expr.getRight()).toString();
		}
		else if (expr.getRight() instanceof DataIdentifier 
			&& ((DataIdentifier)expr.getRight()).getDataType() == DataType.SCALAR
			&& ((DataIdentifier)expr.getRight()).getValueType() == ValueType.STRING){
			String name = ((DataIdentifier)expr.getRight()).getName();
			filename =  filename + ((StringIdentifier)currConstVars.get(name)).getValue();
		}
		else {
			raiseValidateError("Parameter " + IO_FILENAME + " only supports a const string or const string concatenations.", conditional);
		}
		return filename;
			
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(_opcode.toString());
		sb.append("(");

		boolean first = true;
		for(Entry<String,Expression> e : _varParams.entrySet()) {
			String key = e.getKey();
			Expression expr = e.getValue();
			if (!first) {
				sb.append(", ");
			} else {
				first = false;
			}
			sb.append(key);
			sb.append("=");
			if (expr instanceof StringIdentifier) {
				sb.append("\"");
				sb.append(expr);
				sb.append("\"");
			} else {
				sb.append(expr);
			}
		}
		sb.append(")");
		return sb.toString();
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		for( Expression expr : _varParams.values() ) {
			result.addVariables ( expr.variablesRead() );
		}
		return result;
	}

	@Override
	public VariableSet variablesUpdated() {
		VariableSet result = new VariableSet();
		for( Expression expr : _varParams.values() ) {
			result.addVariables ( expr.variablesUpdated() );
		}
		result.addVariable(((DataIdentifier)this.getOutput()).getName(), (DataIdentifier)this.getOutput());
		return result;
	}
	
	public boolean isCSVReadWithUnknownSize() {
		Expression format = getVarParam(FORMAT_TYPE);
		if( _opcode == DataOp.READ && format!=null && checkFormatType(FileFormat.CSV) ) {
			Expression rows = getVarParam(READROWPARAM);
			Expression cols = getVarParam(READCOLPARAM);
			return (rows==null || Long.parseLong(rows.toString())<0)
				||(cols==null || Long.parseLong(cols.toString())<0);
		}
		return false;
	}
	
	public boolean isLIBSVMReadWithUnknownSize() {
		Expression format = getVarParam(FORMAT_TYPE);
		if (_opcode == DataOp.READ && format != null && checkFormatType(FileFormat.LIBSVM)) {
			Expression rows = getVarParam(READROWPARAM);
			Expression cols = getVarParam(READCOLPARAM);
			return (rows == null || Long.parseLong(rows.toString()) < 0) 
				|| (cols == null || Long.parseLong(cols.toString()) < 0);
		}
		return false;
	}
	
	public boolean isRead()
	{
		return (_opcode == DataOp.READ);
	}
} // end class
