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
package org.apache.sysml.api;

import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;

import scala.collection.Seq;

public class MLBlock implements Row {

	private static final long serialVersionUID = -770986277854643424L;

	public MatrixIndexes indexes;
	public MatrixBlock block;
	
	public MLBlock(MatrixIndexes indexes, MatrixBlock block) {
		this.indexes = indexes;
		this.block = block;
	}
	
	@Override
	public boolean anyNull() {
		// TODO
		return false;
	}

	@Override
	public Object apply(int arg0) {
		if(arg0 == 0) {
			return indexes;
		}
		else if(arg0 == 1) {
			return block;
		}
		// TODO: For now not supporting any operations
		return 0;
	}

	@Override
	public Row copy() {
		return new MLBlock(new MatrixIndexes(indexes), new MatrixBlock(block));
	}

	@Override
	public Object get(int arg0) {
		if(arg0 == 0) {
			return indexes;
		}
		else if(arg0 == 1) {
			return block;
		}
		// TODO: For now not supporting any operations
		return 0;
	}

	@Override
	public <T> T getAs(int arg0) {
		// TODO 
		return null;
	}

	@Override
	public <T> T getAs(String arg0) {
		// TODO
		return null;
	}

	@Override
	public boolean getBoolean(int arg0) {
		// TODO
		return false;
	}

	@Override
	public byte getByte(int arg0) {
		// TODO
		return 0;
	}

	@Override
	public Date getDate(int arg0) {
		// TODO
		return null;
	}

	@Override
	public BigDecimal getDecimal(int arg0) {
		// TODO
		return null;
	}

	@Override
	public double getDouble(int arg0) {
		// TODO 
		return 0;
	}

	@Override
	public float getFloat(int arg0) {
		// TODO 
		return 0;
	}

	@Override
	public int getInt(int arg0) {
		// TODO 
		return 0;
	}

	@Override
	public <K, V> Map<K, V> getJavaMap(int arg0) { 
		return null;
	}

	@SuppressWarnings("unchecked")
	@Override
	public <T> List<T> getList(int arg0) {
		ArrayList<Object> retVal = new ArrayList<Object>();
		retVal.add(indexes);
		retVal.add(block);
		//retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(indexes, block));
		return (List<T>) scala.collection.JavaConversions.asScalaBuffer(retVal).toList();
	}

	@Override
	public long getLong(int arg0) {
		// TODO 
		return 0;
	}

	@Override
	public int fieldIndex(String arg0) {
		// TODO
		return 0;
	}

	@Override
	public <K, V> scala.collection.Map<K, V> getMap(int arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public <T> scala.collection.immutable.Map<String, T> getValuesMap(Seq<String> arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@SuppressWarnings("unchecked")
	@Override
	public <T> Seq<T> getSeq(int arg0) {
		ArrayList<Object> retVal = new ArrayList<Object>();
		retVal.add(indexes);
		retVal.add(block);
		// retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(indexes, block));
		return (Seq<T>) scala.collection.JavaConversions.asScalaBuffer(retVal).toSeq();
	}

	@Override
	public short getShort(int arg0) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String getString(int arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Row getStruct(int arg0) {
		return this;
	}

	@Override
	public boolean isNullAt(int arg0) {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int length() {
		return 2;
	}

	@Override
	public String mkString() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String mkString(String arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String mkString(String arg0, String arg1, String arg2) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public StructType schema() {
		return getDefaultSchemaForBinaryBlock();
	}


	@Override
	public int size() {
		return 2;
	}

	@Override
	public Seq<Object> toSeq() {
		ArrayList<Object> retVal = new ArrayList<Object>();
		retVal.add(indexes);
		retVal.add(block);
		// retVal.add(new Tuple2<MatrixIndexes, MatrixBlock>(indexes, block));
		return scala.collection.JavaConversions.asScalaBuffer(retVal).toSeq();
	}
	
	public static StructType getDefaultSchemaForBinaryBlock() {
		// TODO:
		StructField[] fields = new StructField[2];
		fields[0] = new StructField("IgnoreSchema", DataType.fromJson("DoubleType"), true, null);
		fields[1] = new StructField("IgnoreSchema1", DataType.fromJson("DoubleType"), true, null);
		return new StructType(fields);
	}

	@Override
	public Timestamp getTimestamp(int arg0) {
		// TODO Auto-generated method stub
		return null;
	}


}
