package org.apache.sysds.runtime.meta;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.apache.spark.sql.sources.In;
import org.apache.sysds.common.Types;
import org.apache.sysds.parser.DoubleIdentifier;
import org.apache.sysds.parser.Expression;
import org.apache.sysds.parser.StringIdentifier;
import org.scalactic.Bool;

public class MetaDataAll {
	//format
	//data type
	//properties (delim)
	//privacy constraints

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


	public String _dims;
	public double _min = 0;
	public double _max = 1;
	public double _sparsity = 1;
	public int _seed = -1;
	public String _pdf = "uniform";
	public double _lambda = 1;

	// for mtd
	public MatrixCharacteristics _matrixCharacteristics = new MatrixCharacteristics(); // or data characteristics
	public long _dim1 = -1;
	public long _dim2 = -1;

	//csv
	private String _delimiter = null;

	public String _format;

	public int _rowsInBlock;
	public int _colsInBlock;

	public Types.DataType _dataType;
	public Types.ValueType _valueType;
	public String _schema;

	public boolean _header;
	public boolean _sparse;
	public boolean _fill;

	public String _privacy;
	private HashMap<String, Expression> _varParams;

	// TODO are necessary
//	public static final String RAND_BY_ROW = "byrow";
//	public static final String RAND_DIMNAMES = "dimnames";
//	public static final String RAND_DATA = "data";
//
//	// TODO are necessary
//	public static final String SQL_CONN = "conn";
//	public static final String SQL_USER = "user";
//	public static final String SQL_PASS = "password";
//	public static final String SQL_QUERY = "query";
//
//	// TODO are necessary
//	public static final String FED_ADDRESSES = "addresses";
//	public static final String FED_RANGES = "ranges";
//	public static final String FED_TYPE = "type";

	// TODO can be JSONObject
	public static final String PRIVACY = "privacy";
	public static final String FINE_GRAINED_PRIVACY = "fine_grained_privacy";
	public static final String DELIM_NA_STRINGS = "naStrings";


	public MetaDataAll(String format, String dataType, String valueType, String delimiter, int nnz, int rows, int cols) {
		_format = format;
		_dataType = Types.DataType.valueOf(dataType);
		_valueType = Types.ValueType.valueOf(valueType);
		_matrixCharacteristics = new MatrixCharacteristics(rows, cols, nnz);
		_dim1 = rows;
		_dim2 = cols;
		_delimiter = delimiter;
	}

	public MetaDataAll(BufferedReader br) {
		String line;

		try {
			while ((line = br.readLine()) != null) {
				if(!line.contains(":"))
					continue;

				// split the line by :
				String[] parts = line.replace(",", "").replace(" ", "")
					.replace("\"", "").split(":");
				initParam(parts[0].trim(), parts[1].trim());
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

	private void initParam(String key, String value) {
		switch(key) {
			case "dims": _dims = value;

			case "rows": _matrixCharacteristics.setRows(Long.parseLong(value)); _dim1 = Long.parseLong(value); break;
			case "cols": _matrixCharacteristics.setCols(Long.parseLong(value)); _dim2 = Long.parseLong(value); break;
			case "nnz": _matrixCharacteristics.setNonZeros(Long.parseLong(value)); break;

			case "min": _min = Double.parseDouble(value); break;
			case "max": _max = Double.parseDouble(value); break;
			case "sparsity": _sparsity = Double.parseDouble(value); break;
			case "seed": _seed = Integer.parseInt(value); break;
			case "pdf": _pdf = value; break;
			case "lambda": _lambda = Double.parseDouble(value); break;

			case "value_type": _valueType = Types.ValueType.fromExternalString(value.toUpperCase());; break;
			case "data_type": _dataType = Types.DataType.valueOf(value.toUpperCase()); break;
			case "format": _format = value; break;
			case "privacy": _privacy = value;

			case "rows_in_block": _rowsInBlock = Integer.parseInt(value); break;
			case "cols_in_block": _colsInBlock = Integer.parseInt(value); break;
			case "schema": _schema = value; break;
			case "header": _header = Boolean.valueOf(value); break;
			case "sparse": _sparse = Boolean.valueOf(value); break;
			case "fill" : _fill = Boolean.valueOf(value); break;
			case "delimiter": _delimiter = value;
			default: break;
		}

		//TODO _min, _max Rand DoubleIdentifier and add varParams
//		_varParams.put(key, new StringIdentifier(value, ParseInfo));
	}

	//TODO
	public HashMap<String, String> parseVarParams() {
		return null;
	}
}
