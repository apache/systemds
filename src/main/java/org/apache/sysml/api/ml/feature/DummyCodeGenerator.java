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

package org.apache.sysml.api.ml.feature;

import java.util.HashMap;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.attribute.AttributeGroup;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.util.SchemaUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.StructType;

import org.apache.sysml.api.ml.functions.AddMinMaxLabelValue;
import org.apache.sysml.api.ml.functions.ConvertListToRow;
import org.apache.sysml.api.ml.functions.ConvertRowToListOfObjects;
import org.apache.sysml.api.ml.functions.GenerateDummyCode;

public class DummyCodeGenerator extends Transformer implements HasInputCol, HasOutputCol {

	private static final long serialVersionUID = 6837312049351179402L;
	private static double minLabelVal, maxLabelVal;

	private HashMap<String, String> params = new HashMap<String, String>();
	private Param<String> inputCol = new Param<String>(this, "inputCol", "Input column name");
	private Param<String> outputCol = new Param<String>(this, "outputCol", "Output column name");

	public DummyCodeGenerator() {
		setInputCol("label");
		setOutputCol("DummyCodedLabel");
	}

//	public DummyCodeGenerator(String strIn) {
//		this();
//		System.out.println("INPUT PARAM = " + strIn);
//	}

	public static double getMinLabelVal() {
		return minLabelVal;
	}

	public static double getMaxLabelVal() {
		return maxLabelVal;
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public Transformer copy(ParamMap extra) {
		// return defaultCopy(extra);
		DummyCodeGenerator dc = new DummyCodeGenerator();
		dc.setInputCol(getInputCol());
		dc.setOutputCol(getOutputCol());
		return dc;
	}

	public DummyCodeGenerator setInputCol(String value) {
		params.put(inputCol.name(), value);
		return (DummyCodeGenerator) setDefault(inputCol, value);
	}

	@Override
	public Param<String> inputCol() {
		return inputCol;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasInputCol$_setter_$inputCol_$eq(Param arg0) {

	}

	@Override
	public String getInputCol() {
		return params.get(inputCol.name());
	}

	public DummyCodeGenerator setOutputCol(String value) {
		params.put(outputCol.name(), value);
		return (DummyCodeGenerator) setDefault(outputCol, value);
	}

	@Override
	public Param<String> outputCol() {
		return outputCol;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasOutputCol$_setter_$outputCol_$eq(Param arg0) {

	}

	@Override
	public String getOutputCol() {
		return params.get(outputCol.name());
	}

	@Override
	public DataFrame transform(DataFrame dataset) {
		String labelCol = getInputCol();
		minLabelVal = dataset.groupBy().min(labelCol).first().getDouble(0);
		maxLabelVal = dataset.groupBy().max(labelCol).first().getDouble(0);
		StructType outputSchema = transformSchema(dataset.schema());

		JavaRDD<List<Object>> objRDD = dataset.rdd().toJavaRDD().map(new ConvertRowToListOfObjects());
		JavaRDD<List<Object>> addMinMaxVal = objRDD.map(new AddMinMaxLabelValue());
		JavaRDD<List<Object>> dummyCoded = addMinMaxVal.map(new GenerateDummyCode());
		JavaRDD<Row> row = dummyCoded.map(new ConvertListToRow());
		DataFrame result = dataset.sqlContext().createDataFrame(row, outputSchema);

		return result;
	}

	@Override
	public StructType transformSchema(StructType schema) {
		DataType inputType = schema.apply(getInputCol()).dataType();

		if (!(inputType instanceof DoubleType))
			System.err.println("The input column must be DoubleType, but got " + inputType);

		AttributeGroup attrGroup =
				new AttributeGroup(getOutputCol(), (int) (maxLabelVal - minLabelVal + 1));
		return SchemaUtils.appendColumn(schema, attrGroup.toStructField());
	}
}
