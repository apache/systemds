/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.api.mlcontext;

import java.io.File;
import java.io.FileNotFoundException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.tugraz.sysds.conf.CompilerConfig;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.conf.CompilerConfig.ConfigType;
import org.tugraz.sysds.parser.ParseException;
import org.tugraz.sysds.parser.Statement;
import org.tugraz.sysds.runtime.controlprogram.BasicProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.ForProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.IfProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.ProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.WhileProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.caching.FrameObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.BooleanObject;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.DoubleObject;
import org.tugraz.sysds.runtime.instructions.cp.IntObject;
import org.tugraz.sysds.runtime.instructions.cp.StringObject;
import org.tugraz.sysds.runtime.instructions.cp.VariableCPInstruction;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.utils.MLContextProxy;
import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Utility class containing methods for working with the MLContext API.
 *
 */
public final class MLContextUtil {

	/**
	 * Version not available message.
	 */
	public static final String VERSION_NOT_AVAILABLE = "Version not available";

	/**
	 * Build time not available message.
	 */
	public static final String BUILD_TIME_NOT_AVAILABLE = "Build time not available";

	/**
	 * Basic data types supported by the MLContext API.
	 */
	@SuppressWarnings("rawtypes")
	public static final Class[] BASIC_DATA_TYPES = { Integer.class, Boolean.class, Double.class, String.class };

	/**
	 * Complex data types supported by the MLContext API.
	 */
	@SuppressWarnings("rawtypes")
	public static final Class[] COMPLEX_DATA_TYPES = { JavaRDD.class, RDD.class, Dataset.class, Matrix.class,
			Frame.class, (new double[][] {}).getClass(), MatrixBlock.class, URL.class };

	/**
	 * All data types supported by the MLContext API.
	 */
	@SuppressWarnings("rawtypes")
	public static final Class[] ALL_SUPPORTED_DATA_TYPES = (Class[]) ArrayUtils.addAll(BASIC_DATA_TYPES,
			COMPLEX_DATA_TYPES);

	/**
	 * Compare two version strings (ie, "1.4.0" and "1.4.1").
	 *
	 * @param versionStr1
	 *            First version string.
	 * @param versionStr2
	 *            Second version string.
	 * @return If versionStr1 is less than versionStr2, return {@code -1}. If
	 *         versionStr1 equals versionStr2, return {@code 0}. If versionStr1
	 *         is greater than versionStr2, return {@code 1}.
	 * @throws MLContextException
	 *             if versionStr1 or versionStr2 is {@code null}
	 */
	private static int compareVersion(String versionStr1, String versionStr2) {
		if (versionStr1 == null) {
			throw new MLContextException("First version argument to compareVersion() is null");
		}
		if (versionStr2 == null) {
			throw new MLContextException("Second version argument to compareVersion() is null");
		}

		Scanner scanner1 = null;
		Scanner scanner2 = null;
		try {
			scanner1 = new Scanner(versionStr1);
			scanner2 = new Scanner(versionStr2);
			scanner1.useDelimiter("\\.");
			scanner2.useDelimiter("\\.");

			while (scanner1.hasNextInt() && scanner2.hasNextInt()) {
				int version1 = scanner1.nextInt();
				int version2 = scanner2.nextInt();
				if (version1 < version2) {
					return -1;
				} else if (version1 > version2) {
					return 1;
				}
			}

			return scanner1.hasNextInt() ? 1 : 0;
		} finally {
			scanner1.close();
			scanner2.close();
		}
	}

	/**
	 * Determine whether the Spark version is supported.
	 *
	 * @param sparkVersion
	 *            Spark version string (ie, "1.5.0").
	 * @param minimumRecommendedSparkVersion
	 *            Minimum recommended Spark version string (ie, "2.1.0").
	 * @return {@code true} if Spark version supported; otherwise {@code false}.
	 */
	public static boolean isSparkVersionSupported(String sparkVersion, String minimumRecommendedSparkVersion) {
		return compareVersion(sparkVersion, minimumRecommendedSparkVersion) >= 0;
	}

	/**
	 * Check that the Spark version is supported. If it isn't supported, throw
	 * an MLContextException.
	 *
	 * @param spark
	 *            SparkSession
	 * @throws MLContextException
	 *             thrown if Spark version isn't supported
	 */
	public static void verifySparkVersionSupported(SparkSession spark) {
		String minimumRecommendedSparkVersion = null;
		try {
			// If this is being called using the SystemDS jar file,
			// ProjectInfo should be available.
			ProjectInfo projectInfo = ProjectInfo.getProjectInfo();
			minimumRecommendedSparkVersion = projectInfo.minimumRecommendedSparkVersion();
		} catch (MLContextException e) {
			try {
				// During development (such as in an IDE), there is no jar file
				// typically
				// built, so attempt to obtain the minimum recommended Spark
				// version from
				// the pom.xml file
				minimumRecommendedSparkVersion = getMinimumRecommendedSparkVersionFromPom();
			} catch (MLContextException e1) {
				throw new MLContextException(
						"Minimum recommended Spark version could not be determined from SystemDS jar file manifest or pom.xml");
			}
		}
		String sparkVersion = spark.version();
		if (!MLContextUtil.isSparkVersionSupported(sparkVersion, minimumRecommendedSparkVersion)) {
			throw new MLContextException(
					"Spark " + sparkVersion + " or greater is recommended for this version of SystemDS.");
		}
	}

	/**
	 * Obtain minimum recommended Spark version from the pom.xml file.
	 *
	 * @return the minimum recommended Spark version from XML parsing of the pom
	 *         file (during development).
	 */
	static String getMinimumRecommendedSparkVersionFromPom() {
		return getUniquePomProperty("spark.version");
	}

	/**
	 * Obtain the text associated with an XML element from the pom.xml file. In
	 * this implementation, the element should be uniquely named, or results
	 * will be unpredicable.
	 *
	 * @param property
	 *            unique property (element) from the pom.xml file
	 * @return the text value associated with the given property
	 */
	static String getUniquePomProperty(String property) {
		File f = new File("pom.xml");
		if (!f.exists()) {
			throw new MLContextException("pom.xml not found");
		}
		try {
			DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
			DocumentBuilder builder = dbf.newDocumentBuilder();
			Document document = builder.parse(f);

			NodeList nodes = document.getElementsByTagName(property);
			int length = nodes.getLength();
			if (length == 0) {
				throw new MLContextException("Property not found in pom.xml");
			}
			Node node = nodes.item(0);
			String value = node.getTextContent();
			return value;
		} catch (Exception e) {
			throw new MLContextException("MLContextException when reading property '" + property + "' from pom.xml", e);
		}
	}

	/**
	 * Set default SystemDS configuration properties.
	 */
	public static void setDefaultConfig() {
		ConfigurationManager.setGlobalConfig(new DMLConfig());
	}

	/**
	 * Set SystemDS configuration properties based on a configuration file.
	 *
	 * @param configFilePath
	 *            Path to configuration file.
	 * @throws MLContextException
	 *             if configuration file was not found or a parse exception
	 *             occurred
	 */
	public static void setConfig(String configFilePath) {
		try {
			DMLConfig config = new DMLConfig(configFilePath);
			ConfigurationManager.setGlobalConfig(config);
		} catch (ParseException e) {
			throw new MLContextException("Parse Exception when setting config", e);
		} catch (FileNotFoundException e) {
			throw new MLContextException("File not found (" + configFilePath + ") when setting config", e);
		}
	}

	/**
	 * Set SystemDS compiler configuration properties for MLContext
	 */
	public static void setCompilerConfig() {
		CompilerConfig compilerConfig = new CompilerConfig();
		compilerConfig.set(ConfigType.IGNORE_UNSPECIFIED_ARGS, true);
		compilerConfig.set(ConfigType.REJECT_READ_WRITE_UNKNOWNS, false);
		compilerConfig.set(ConfigType.ALLOW_CSE_PERSISTENT_READS, false);
		compilerConfig.set(ConfigType.MLCONTEXT, true);
		ConfigurationManager.setGlobalConfig(compilerConfig);
	}

	/**
	 * Verify that the types of input values are supported.
	 *
	 * @param inputs
	 *            Map of String/Object pairs
	 * @throws MLContextException
	 *             if an input value type is not supported
	 */
	public static void checkInputValueTypes(Map<String, Object> inputs) {
		for (Entry<String, Object> entry : inputs.entrySet()) {
			checkInputValueType(entry.getKey(), entry.getValue());
		}
	}

	/**
	 * Verify that the type of input value is supported.
	 *
	 * @param name
	 *            The name of the input
	 * @param value
	 *            The value of the input
	 * @throws MLContextException
	 *             if the input value type is not supported
	 */
	public static void checkInputValueType(String name, Object value) {

		if (name == null) {
			throw new MLContextException("No input name supplied");
		} else if (value == null) {
			throw new MLContextException("No input value supplied");
		}

		Object o = value;
		boolean supported = false;
		for (Class<?> clazz : ALL_SUPPORTED_DATA_TYPES) {
			if (o.getClass().equals(clazz)) {
				supported = true;
				break;
			} else if (clazz.isAssignableFrom(o.getClass())) {
				supported = true;
				break;
			}
		}
		if (!supported) {
			throw new MLContextException("Input name (\"" + name + "\") value type not supported: " + o.getClass());
		}
	}

	/**
	 * Verify that the type of input parameter value is supported.
	 *
	 * @param parameterName
	 *            The name of the input parameter
	 * @param parameterValue
	 *            The value of the input parameter
	 * @throws MLContextException
	 *             if the input parameter value type is not supported
	 */
	public static void checkInputParameterType(String parameterName, Object parameterValue) {

		if (parameterName == null) {
			throw new MLContextException("No parameter name supplied");
		} else if (parameterValue == null) {
			throw new MLContextException("No parameter value supplied");
		} else if (!parameterName.startsWith("$")) {
			throw new MLContextException("Input parameter name must start with a $");
		}

		Object o = parameterValue;
		boolean supported = false;
		for (Class<?> clazz : BASIC_DATA_TYPES) {
			if (o.getClass().equals(clazz)) {
				supported = true;
				break;
			} else if (clazz.isAssignableFrom(o.getClass())) {
				supported = true;
				break;
			}
		}
		if (!supported) {
			throw new MLContextException(
					"Input parameter (\"" + parameterName + "\") value type not supported: " + o.getClass());
		}
	}

	/**
	 * Is the object one of the supported basic data types? (Integer, Boolean,
	 * Double, String)
	 *
	 * @param object
	 *            the object type to be examined
	 * @return {@code true} if type is a basic data type; otherwise
	 *         {@code false}.
	 */
	public static boolean isBasicType(Object object) {
		for (Class<?> clazz : BASIC_DATA_TYPES) {
			if (object.getClass().equals(clazz)) {
				return true;
			} else if (clazz.isAssignableFrom(object.getClass())) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Obtain the SystemDS scalar value type string equivalent of an accepted
	 * basic type (Integer, Boolean, Double, String)
	 *
	 * @param object
	 *            the object type to be examined
	 * @return a String representing the type as a SystemDS scalar value type
	 */
	public static String getBasicTypeString(Object object) {
		if (!isBasicType(object)) {
			throw new MLContextException("Type (" + object.getClass() + ") not a recognized basic type");
		}
		Class<? extends Object> clazz = object.getClass();
		if (clazz.equals(Integer.class)) {
			return Statement.INT_VALUE_TYPE;
		} else if (clazz.equals(Boolean.class)) {
			return Statement.BOOLEAN_VALUE_TYPE;
		} else if (clazz.equals(Double.class)) {
			return Statement.DOUBLE_VALUE_TYPE;
		} else if (clazz.equals(String.class)) {
			return Statement.STRING_VALUE_TYPE;
		} else {
			return null;
		}
	}

	/**
	 * Is the object one of the supported complex data types? (JavaRDD, RDD,
	 * DataFrame, Matrix, double[][], MatrixBlock, URL)
	 *
	 * @param object
	 *            the object type to be examined
	 * @return {@code true} if type is a complex data type; otherwise
	 *         {@code false}.
	 */
	public static boolean isComplexType(Object object) {
		for (Class<?> clazz : COMPLEX_DATA_TYPES) {
			if (object.getClass().equals(clazz)) {
				return true;
			} else if (clazz.isAssignableFrom(object.getClass())) {
				return true;
			}
		}
		return false;
	}

	/**
	 * Converts non-string basic input parameter values to strings to pass to
	 * the parser.
	 *
	 * @param basicInputParameterMap
	 *            map of input parameters
	 * @return map of String/String name/value pairs
	 */
	public static Map<String, String> convertInputParametersForParser(Map<String, Object> basicInputParameterMap) {
		if (basicInputParameterMap == null) {
			return null;
		}
		Map<String, String> convertedMap = new HashMap<>();
		for (Entry<String, Object> entry : basicInputParameterMap.entrySet()) {
			String key = entry.getKey();
			Object value = entry.getValue();
			if (value == null) {
				throw new MLContextException("Input parameter value is null for: " + entry.getKey());
			} else if (value instanceof Integer) {
				convertedMap.put(key, Integer.toString((Integer) value));
			} else if (value instanceof Boolean) {
				convertedMap.put(key, String.valueOf((Boolean) value).toUpperCase());
			} else if (value instanceof Double) {
				convertedMap.put(key, Double.toString((Double) value));
			} else if (value instanceof String) {
				convertedMap.put(key, (String) value);
			} else {
				throw new MLContextException("Incorrect type for input parameters");
			}
		}
		return convertedMap;
	}

	/**
	 * Convert input types to internal SystemDS representations
	 *
	 * @param parameterName
	 *            The name of the input parameter
	 * @param parameterValue
	 *            The value of the input parameter
	 * @return input in SystemDS data representation
	 */
	public static Data convertInputType(String parameterName, Object parameterValue) {
		return convertInputType(parameterName, parameterValue, null);
	}

	/**
	 * Convert input types to internal SystemDS representations
	 *
	 * @param parameterName
	 *            The name of the input parameter
	 * @param parameterValue
	 *            The value of the input parameter
	 * @param metadata
	 *            matrix/frame metadata
	 * @return input in SystemDS data representation
	 */
	public static Data convertInputType(String parameterName, Object parameterValue, Metadata metadata) {
		String name = parameterName;
		Object value = parameterValue;
		boolean hasMetadata = (metadata != null) ? true : false;
		boolean hasMatrixMetadata = hasMetadata && (metadata instanceof MatrixMetadata) ? true : false;
		boolean hasFrameMetadata = hasMetadata && (metadata instanceof FrameMetadata) ? true : false;
		if (name == null) {
			throw new MLContextException("Input parameter name is null");
		} else if (value == null) {
			throw new MLContextException("Input parameter value is null for: " + parameterName);
		} else if (value instanceof JavaRDD<?>) {
			@SuppressWarnings("unchecked")
			JavaRDD<String> javaRDD = (JavaRDD<String>) value;

			if (hasMatrixMetadata) {
				MatrixMetadata matrixMetadata = (MatrixMetadata) metadata;
				if (matrixMetadata.getMatrixFormat() == MatrixFormat.IJV) {
					return MLContextConversionUtil.javaRDDStringIJVToMatrixObject(javaRDD, matrixMetadata);
				} else {
					return MLContextConversionUtil.javaRDDStringCSVToMatrixObject(javaRDD, matrixMetadata);
				}
			} else if (hasFrameMetadata) {
				FrameMetadata frameMetadata = (FrameMetadata) metadata;
				if (frameMetadata.getFrameFormat() == FrameFormat.IJV) {
					return MLContextConversionUtil.javaRDDStringIJVToFrameObject(javaRDD, frameMetadata);
				} else {
					return MLContextConversionUtil.javaRDDStringCSVToFrameObject(javaRDD, frameMetadata);
				}
			} else if (!hasMetadata) {
				String firstLine = javaRDD.first();
				boolean isAllNumbers = isCSVLineAllNumbers(firstLine);
				if (isAllNumbers) {
					return MLContextConversionUtil.javaRDDStringCSVToMatrixObject(javaRDD);
				} else {
					return MLContextConversionUtil.javaRDDStringCSVToFrameObject(javaRDD);
				}
			}

		} else if (value instanceof RDD<?>) {
			@SuppressWarnings("unchecked")
			RDD<String> rdd = (RDD<String>) value;

			if (hasMatrixMetadata) {
				MatrixMetadata matrixMetadata = (MatrixMetadata) metadata;
				if (matrixMetadata.getMatrixFormat() == MatrixFormat.IJV) {
					return MLContextConversionUtil.rddStringIJVToMatrixObject(rdd, matrixMetadata);
				} else {
					return MLContextConversionUtil.rddStringCSVToMatrixObject(rdd, matrixMetadata);
				}
			} else if (hasFrameMetadata) {
				FrameMetadata frameMetadata = (FrameMetadata) metadata;
				if (frameMetadata.getFrameFormat() == FrameFormat.IJV) {
					return MLContextConversionUtil.rddStringIJVToFrameObject(rdd, frameMetadata);
				} else {
					return MLContextConversionUtil.rddStringCSVToFrameObject(rdd, frameMetadata);
				}
			} else if (!hasMetadata) {
				String firstLine = rdd.first();
				boolean isAllNumbers = isCSVLineAllNumbers(firstLine);
				if (isAllNumbers) {
					return MLContextConversionUtil.rddStringCSVToMatrixObject(rdd);
				} else {
					return MLContextConversionUtil.rddStringCSVToFrameObject(rdd);
				}
			}
		} else if (value instanceof MatrixBlock) {
			MatrixBlock matrixBlock = (MatrixBlock) value;
			return MLContextConversionUtil.matrixBlockToMatrixObject(name, matrixBlock, (MatrixMetadata) metadata);
		} else if (value instanceof FrameBlock) {
			FrameBlock frameBlock = (FrameBlock) value;
			return MLContextConversionUtil.frameBlockToFrameObject(name, frameBlock, (FrameMetadata) metadata);
		} else if (value instanceof Dataset<?>) {
			@SuppressWarnings("unchecked")
			Dataset<Row> dataFrame = (Dataset<Row>) value;

			dataFrame = MLUtils.convertVectorColumnsToML(dataFrame);
			if (hasMatrixMetadata) {
				return MLContextConversionUtil.dataFrameToMatrixObject(dataFrame, (MatrixMetadata) metadata);
			} else if (hasFrameMetadata) {
				return MLContextConversionUtil.dataFrameToFrameObject(dataFrame, (FrameMetadata) metadata);
			} else if (!hasMetadata) {
				boolean looksLikeMatrix = doesDataFrameLookLikeMatrix(dataFrame);
				if (looksLikeMatrix) {
					return MLContextConversionUtil.dataFrameToMatrixObject(dataFrame);
				} else {
					return MLContextConversionUtil.dataFrameToFrameObject(dataFrame);
				}
			}
		} else if (value instanceof Matrix) {
			Matrix matrix = (Matrix) value;
			if ((matrix.hasBinaryBlocks()) && (!matrix.hasMatrixObject())) {
				if (metadata == null) {
					metadata = matrix.getMatrixMetadata();
				}
				JavaPairRDD<MatrixIndexes, MatrixBlock> binaryBlocks = matrix.toBinaryBlocks();
				return MLContextConversionUtil.binaryBlocksToMatrixObject(binaryBlocks,
						(MatrixMetadata) metadata);
			} else {
				return matrix.toMatrixObject();
			}
		} else if (value instanceof Frame) {
			Frame frame = (Frame) value;
			if ((frame.hasBinaryBlocks()) && (!frame.hasFrameObject())) {
				if (metadata == null) {
					metadata = frame.getFrameMetadata();
				}
				JavaPairRDD<Long, FrameBlock> binaryBlocks = frame.toBinaryBlocks();
				return MLContextConversionUtil.binaryBlocksToFrameObject(binaryBlocks, (FrameMetadata) metadata);
			} else {
				return frame.toFrameObject();
			}
		} else if (value instanceof double[][]) {
			double[][] doubleMatrix = (double[][]) value;
			return MLContextConversionUtil.doubleMatrixToMatrixObject(name, doubleMatrix, (MatrixMetadata) metadata);
		} else if (value instanceof URL) {
			URL url = (URL) value;
			return MLContextConversionUtil.urlToMatrixObject(url, (MatrixMetadata) metadata);
		} else if (value instanceof Integer) {
			return new IntObject((Integer) value);
		} else if (value instanceof Double) {
			return new DoubleObject((Double) value);
		} else if (value instanceof String) {
			return new StringObject((String) value);
		} else if (value instanceof Boolean) {
			return new BooleanObject((Boolean) value);
		}
		return null;
	}

	/**
	 * If no metadata is supplied for an RDD or JavaRDD, this method can be used
	 * to determine whether the data appears to be matrix (or a frame)
	 *
	 * @param line
	 *            a line of the RDD
	 * @return {@code true} if all the csv-separated values are numbers,
	 *         {@code false} otherwise
	 */
	public static boolean isCSVLineAllNumbers(String line) {
		if (StringUtils.isBlank(line)) {
			return false;
		}
		String[] parts = line.split(",");
		for (int i = 0; i < parts.length; i++) {
			String part = parts[i].trim();
			try {
				Double.parseDouble(part);
			} catch (NumberFormatException e) {
				return false;
			}
		}
		return true;
	}

	/**
	 * Examine the DataFrame schema to determine whether the data appears to be
	 * a matrix.
	 *
	 * @param df
	 *            the DataFrame
	 * @return {@code true} if the DataFrame appears to be a matrix,
	 *         {@code false} otherwise
	 */
	public static boolean doesDataFrameLookLikeMatrix(Dataset<Row> df) {
		StructType schema = df.schema();
		StructField[] fields = schema.fields();
		if (fields == null) {
			return true;
		}
		for (StructField field : fields) {
			DataType dataType = field.dataType();
			if ((dataType != DataTypes.DoubleType) && (dataType != DataTypes.IntegerType)
					&& (dataType != DataTypes.LongType) && (!(dataType instanceof org.apache.spark.ml.linalg.VectorUDT))
					&& (!(dataType instanceof org.apache.spark.mllib.linalg.VectorUDT))) {
				// uncomment if we support arrays of doubles for matrices
				// if (dataType instanceof ArrayType) {
				// ArrayType arrayType = (ArrayType) dataType;
				// if (arrayType.elementType() == DataTypes.DoubleType) {
				// continue;
				// }
				// }
				return false;
			}
		}
		return true;
	}

	/**
	 * Return a double-quoted string with inner single and double quotes
	 * escaped.
	 *
	 * @param str
	 *            the original string
	 * @return double-quoted string with inner single and double quotes escaped
	 */
	public static String quotedString(String str) {
		if (str == null) {
			return null;
		}

		StringBuilder sb = new StringBuilder();
		sb.append("\"");
		for (int i = 0; i < str.length(); i++) {
			char ch = str.charAt(i);
			if ((ch == '\'') || (ch == '"')) {
				if ((i > 0) && (str.charAt(i - 1) != '\\')) {
					sb.append('\\');
				} else if (i == 0) {
					sb.append('\\');
				}
			}
			sb.append(ch);
		}
		sb.append("\"");

		return sb.toString();
	}

	/**
	 * Display the keys and values in a Map
	 *
	 * @param mapName
	 *            the name of the map
	 * @param map
	 *            Map of String keys and Object values
	 * @return the keys and values in the Map as a String
	 */
	public static String displayMap(String mapName, Map<String, Object> map) {
		StringBuilder sb = new StringBuilder();
		sb.append(mapName);
		sb.append(":\n");
		Set<String> keys = map.keySet();
		if (keys.isEmpty()) {
			sb.append("None\n");
		} else {
			int count = 0;
			for (String key : keys) {
				sb.append("  [");
				sb.append(++count);
				sb.append("] ");
				sb.append(key);
				sb.append(": ");
				sb.append(map.get(key));
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	/**
	 * Display the values in a Set
	 *
	 * @param setName
	 *            the name of the Set
	 * @param set
	 *            Set of String values
	 * @return the values in the Set as a String
	 */
	public static String displaySet(String setName, Set<String> set) {
		StringBuilder sb = new StringBuilder();
		sb.append(setName);
		sb.append(":\n");
		if (set.isEmpty()) {
			sb.append("None\n");
		} else {
			int count = 0;
			for (String value : set) {
				sb.append("  [");
				sb.append(++count);
				sb.append("] ");
				sb.append(value);
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	/**
	 * Display the keys and values in the symbol table
	 *
	 * @param name
	 *            the name of the symbol table
	 * @param symbolTable
	 *            the LocalVariableMap
	 * @return the keys and values in the symbol table as a String
	 */
	public static String displaySymbolTable(String name, LocalVariableMap symbolTable) {
		StringBuilder sb = new StringBuilder();
		sb.append(name);
		sb.append(":\n");
		sb.append(displaySymbolTable(symbolTable));
		return sb.toString();
	}

	/**
	 * Display the keys and values in the symbol table
	 *
	 * @param symbolTable
	 *            the LocalVariableMap
	 * @return the keys and values in the symbol table as a String
	 */
	public static String displaySymbolTable(LocalVariableMap symbolTable) {
		StringBuilder sb = new StringBuilder();
		Set<String> keys = symbolTable.keySet();
		if (keys.isEmpty()) {
			sb.append("None\n");
		} else {
			int count = 0;
			for (String key : keys) {
				sb.append("  [");
				sb.append(++count);
				sb.append("]");

				sb.append(" (");
				sb.append(determineOutputTypeAsString(symbolTable, key));
				sb.append(") ");

				sb.append(key);

				sb.append(": ");
				sb.append(symbolTable.get(key));
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	/**
	 * Obtain a symbol table output type as a String
	 *
	 * @param symbolTable
	 *            the symbol table
	 * @param outputName
	 *            the name of the output variable
	 * @return the symbol table output type for a variable as a String
	 */
	public static String determineOutputTypeAsString(LocalVariableMap symbolTable, String outputName) {
		Data data = symbolTable.get(outputName);
		if (data instanceof BooleanObject) {
			return "Boolean";
		} else if (data instanceof DoubleObject) {
			return "Double";
		} else if (data instanceof IntObject) {
			return "Long";
		} else if (data instanceof StringObject) {
			return "String";
		} else if (data instanceof MatrixObject) {
			return "Matrix";
		} else if (data instanceof FrameObject) {
			return "Frame";
		}
		return "Unknown";
	}

	/**
	 * Obtain a display of script inputs.
	 *
	 * @param name
	 *            the title to display for the inputs
	 * @param map
	 *            the map of inputs
	 * @param symbolTable
	 *            the symbol table
	 * @return the script inputs represented as a String
	 */
	public static String displayInputs(String name, Map<String, Object> map, LocalVariableMap symbolTable) {
		StringBuilder sb = new StringBuilder();
		sb.append(name);
		sb.append(":\n");
		Set<String> keys = map.keySet();
		if (keys.isEmpty()) {
			sb.append("None\n");
		} else {
			int count = 0;
			for (String key : keys) {
				Object object = map.get(key);
				@SuppressWarnings("rawtypes")
				Class clazz = object.getClass();
				String type = clazz.getSimpleName();
				if (object instanceof JavaRDD<?>) {
					type = "JavaRDD";
				} else if (object instanceof RDD<?>) {
					type = "RDD";
				}

				sb.append("  [");
				sb.append(++count);
				sb.append("]");

				sb.append(" (");
				sb.append(type);
				if (doesSymbolTableContainMatrixObject(symbolTable, key)) {
					sb.append(" as Matrix");
				} else if (doesSymbolTableContainFrameObject(symbolTable, key)) {
					sb.append(" as Frame");
				}
				sb.append(") ");

				sb.append(key);
				sb.append(": ");
				String str = null;
				if (object instanceof MatrixBlock) {
					MatrixBlock mb = (MatrixBlock) object;
					str = "MatrixBlock [sparse? = " + mb.isInSparseFormat() + ", nonzeros = " + mb.getNonZeros()
							+ ", size: " + mb.getNumRows() + " X " + mb.getNumColumns() + "]";
				} else
					str = object.toString(); // TODO: Deal with OOM for other
												// objects such as Frame, etc
				str = StringUtils.abbreviate(str, 100);
				sb.append(str);
				sb.append("\n");
			}
		}
		return sb.toString();
	}

	/**
	 * Obtain a display of the script outputs.
	 *
	 * @param name
	 *            the title to display for the outputs
	 * @param outputNames
	 *            the names of the output variables
	 * @param symbolTable
	 *            the symbol table
	 * @return the script outputs represented as a String
	 *
	 */
	public static String displayOutputs(String name, Set<String> outputNames, LocalVariableMap symbolTable) {
		StringBuilder sb = new StringBuilder();
		sb.append(name);
		sb.append(":\n");
		sb.append(displayOutputs(outputNames, symbolTable));
		return sb.toString();
	}

	/**
	 * Obtain a display of the script outputs.
	 *
	 * @param outputNames
	 *            the names of the output variables
	 * @param symbolTable
	 *            the symbol table
	 * @return the script outputs represented as a String
	 *
	 */
	public static String displayOutputs(Set<String> outputNames, LocalVariableMap symbolTable) {
		StringBuilder sb = new StringBuilder();
		if (outputNames.isEmpty()) {
			sb.append("None\n");
		} else {
			int count = 0;
			for (String outputName : outputNames) {
				sb.append("  [");
				sb.append(++count);
				sb.append("] ");

				if (symbolTable.get(outputName) != null) {
					sb.append("(");
					sb.append(determineOutputTypeAsString(symbolTable, outputName));
					sb.append(") ");
				}

				sb.append(outputName);

				if (symbolTable.get(outputName) != null) {
					sb.append(": ");
					sb.append(symbolTable.get(outputName));
				}

				sb.append("\n");
			}
		}
		return sb.toString();
	}

	/**
	 * The SystemDS welcome message
	 *
	 * @return the SystemDS welcome message
	 */
	public static String welcomeMessage() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nWelcome to Apache SystemDS!\n");
		try {
			ProjectInfo info = ProjectInfo.getProjectInfo();
			if (info.version() != null) {
				sb.append("Version ");
				sb.append(info.version());
			}
		} catch (MLContextException e) {
		}
		return sb.toString();
	}

	/**
	 * Obtain the Spark Context
	 *
	 * @param mlContext
	 *            the SystemDS MLContext
	 * @return the Spark Context
	 */
	public static SparkContext getSparkContext(MLContext mlContext) {
		return mlContext.getSparkSession().sparkContext();
	}

	/**
	 * Obtain the Java Spark Context
	 *
	 * @param mlContext
	 *            the SystemDS MLContext
	 * @return the Java Spark Context
	 */
	public static JavaSparkContext getJavaSparkContext(MLContext mlContext) {
		return new JavaSparkContext(mlContext.getSparkSession().sparkContext());
	}

	/**
	 * Obtain the Spark Context from the MLContextProxy
	 *
	 * @return the Spark Context
	 */
	public static SparkContext getSparkContextFromProxy() {
		MLContext activeMLContext = MLContextProxy.getActiveMLContextForAPI();
		SparkContext sc = getSparkContext(activeMLContext);
		return sc;
	}

	/**
	 * Obtain the Java Spark Context from the MLContextProxy
	 *
	 * @return the Java Spark Context
	 */
	public static JavaSparkContext getJavaSparkContextFromProxy() {
		MLContext activeMLContext = MLContextProxy.getActiveMLContextForAPI();
		JavaSparkContext jsc = getJavaSparkContext(activeMLContext);
		return jsc;
	}

	/**
	 * Obtain the Spark Session from the MLContextProxy
	 *
	 * @return the Spark Session
	 */
	public static SparkSession getSparkSessionFromProxy() {
		return MLContextProxy.getActiveMLContextForAPI().getSparkSession();
	}

	/**
	 * Determine if the symbol table contains a FrameObject with the given
	 * variable name.
	 *
	 * @param symbolTable
	 *            the LocalVariableMap
	 * @param variableName
	 *            the variable name
	 * @return {@code true} if the variable in the symbol table is a
	 *         FrameObject, {@code false} otherwise.
	 */
	public static boolean doesSymbolTableContainFrameObject(LocalVariableMap symbolTable, String variableName) {
		return (symbolTable != null && symbolTable.keySet().contains(variableName)
				&& symbolTable.get(variableName) instanceof FrameObject);
	}

	/**
	 * Determine if the symbol table contains a MatrixObject with the given
	 * variable name.
	 *
	 * @param symbolTable
	 *            the LocalVariableMap
	 * @param variableName
	 *            the variable name
	 * @return {@code true} if the variable in the symbol table is a
	 *         MatrixObject, {@code false} otherwise.
	 */
	public static boolean doesSymbolTableContainMatrixObject(LocalVariableMap symbolTable, String variableName) {
		return (symbolTable != null && symbolTable.keySet().contains(variableName)
				&& symbolTable.get(variableName) instanceof MatrixObject);
	}

	/**
	 * Delete the 'remove variable' instructions from a runtime program.
	 *
	 * @param progam
	 *            runtime program
	 */
	public static void deleteRemoveVariableInstructions(Program progam) {
		Map<String, FunctionProgramBlock> fpbs = progam.getFunctionProgramBlocks();
		if (fpbs != null && !fpbs.isEmpty()) {
			for (Entry<String, FunctionProgramBlock> e : fpbs.entrySet()) {
				FunctionProgramBlock fpb = e.getValue();
				for (ProgramBlock pb : fpb.getChildBlocks()) {
					deleteRemoveVariableInstructions(pb);
				}
			}
		}

		for (ProgramBlock pb : progam.getProgramBlocks()) {
			deleteRemoveVariableInstructions(pb);
		}
	}

	/**
	 * Recursively traverse program block to delete 'remove variable'
	 * instructions.
	 *
	 * @param pb
	 *            Program block
	 */
	private static void deleteRemoveVariableInstructions(ProgramBlock pb) {
		if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock wpb = (WhileProgramBlock) pb;
			for (ProgramBlock pbc : wpb.getChildBlocks())
				deleteRemoveVariableInstructions(pbc);
		} else if (pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock) pb;
			for (ProgramBlock pbc : ipb.getChildBlocksIfBody())
				deleteRemoveVariableInstructions(pbc);
			for (ProgramBlock pbc : ipb.getChildBlocksElseBody())
				deleteRemoveVariableInstructions(pbc);
		} else if (pb instanceof ForProgramBlock) {
			ForProgramBlock fpb = (ForProgramBlock) pb;
			for (ProgramBlock pbc : fpb.getChildBlocks())
				deleteRemoveVariableInstructions(pbc);
		} else if( pb instanceof BasicProgramBlock ) {
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			ArrayList<Instruction> instructions = bpb.getInstructions();
			deleteRemoveVariableInstructions(instructions);
		}
	}

	/**
	 * Delete 'remove variable' instructions.
	 *
	 * @param instructions
	 *            list of instructions
	 */
	private static void deleteRemoveVariableInstructions(ArrayList<Instruction> instructions) {
		for (int i = 0; i < instructions.size(); i++) {
			Instruction linst = instructions.get(i);
			if (linst instanceof VariableCPInstruction && ((VariableCPInstruction) linst).isRemoveVariable()) {
				VariableCPInstruction varinst = (VariableCPInstruction) linst;
				instructions.remove(varinst);
				i--;
			}
		}
	}

}
