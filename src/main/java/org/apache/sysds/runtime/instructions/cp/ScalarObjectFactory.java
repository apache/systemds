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

package org.apache.sysds.runtime.instructions.cp;

import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class ScalarObjectFactory
{
	public static ScalarObject createScalarObject(String value) {
		//best effort parsing of specialized types
		if( UtilFunctions.isBoolean(value) )
			return new BooleanObject(Boolean.parseBoolean(value));
		if( UtilFunctions.isIntegerNumber(value) )
			return new IntObject(UtilFunctions.parseToLong(value));
		return new StringObject(value);
	}
	
	public static ScalarObject createScalarObject(ValueType vt, String value) {
		switch( vt ) {
			case INT64:   return new IntObject(UtilFunctions.parseToLong(value));
			case FP64:    return new DoubleObject(Double.parseDouble(value));
			case BOOLEAN: return new BooleanObject(Boolean.parseBoolean(value));
			case STRING:  return new StringObject(value);
			default: throw new RuntimeException("Unsupported scalar value type: "+vt.name());
		}
	}
	
	public static ScalarObject createScalarObject(ValueType vt, Object obj) {
		//TODO add new scalar object for extended type system
		switch( vt ) {
			case BOOLEAN: return new BooleanObject((Boolean)obj);
			case INT64:   return new IntObject((Long)obj);
			case INT32:   return new IntObject((Integer)obj);
			case FP64:    return new DoubleObject((Double)obj);
			case FP32:    return new DoubleObject((Float)obj);
			case STRING:  return new StringObject((String)obj);
			default: throw new RuntimeException("Unsupported scalar value type: "+vt.name());
		}
	}
	
	public static ScalarObject createScalarObject(ValueType vt, double value) {
		switch( vt ) {
			case INT64:     return new IntObject(UtilFunctions.toLong(value));
			case FP64:  return new DoubleObject(value);
			case BOOLEAN: return new BooleanObject(value != 0);
			case STRING:  return new StringObject(String.valueOf(value));
			default: throw new RuntimeException("Unsupported scalar value type: "+vt.name());
		}
	}
	
	public static ScalarObject createScalarObject(ValueType vt, ScalarObject so) {
		switch( vt ) {
			case FP64:    return castToDouble(so);
			case INT64:   return castToLong(so);
			case BOOLEAN: return new BooleanObject(so.getBooleanValue());
			case STRING:  return new StringObject(so.getStringValue());
			default: throw new RuntimeException("Unsupported scalar value type: "+vt.name());
		}
	}
	
	public static ScalarObject createScalarObject(LiteralOp lit) {
		return createScalarObject(lit.getValueType(), lit);
	}
	
	public static ScalarObject createScalarObject(ValueType vt, LiteralOp lit) {
		switch( vt ) {
			case FP64:    return new DoubleObject(lit.getDoubleValue());
			case INT64:   return new IntObject(lit.getLongValue());
			case BOOLEAN: return new BooleanObject(lit.getBooleanValue());
			case STRING:  return new StringObject(lit.getStringValue());
			default: throw new RuntimeException("Unsupported scalar value type: "+vt.name());
		}
	}
	
	public static LiteralOp createLiteralOp(ScalarObject so) {
		switch( so.getValueType() ){
			case FP64:    return new LiteralOp(so.getDoubleValue());
			case INT64:   return new LiteralOp(so.getLongValue());
			case BOOLEAN: return new LiteralOp(so.getBooleanValue());
			case STRING:  return new LiteralOp(so.getStringValue());
			default:
				throw new HopsException("Unsupported literal value type: "+so.getValueType());
		}
	}
	
	public static LiteralOp createLiteralOp(ScalarObject so, UnaryOp cast) {
		switch( cast.getOp() ) {
			case CAST_AS_DOUBLE:  return new LiteralOp(castToDouble(so).getDoubleValue());
			case CAST_AS_INT:     return new LiteralOp(castToLong(so).getLongValue());
			case CAST_AS_BOOLEAN: return new LiteralOp(so.getBooleanValue());
			default: return null; //otherwise: do nothing
		}
	}
	
	public static LiteralOp createLiteralOp(ValueType vt, String value) {
		switch( vt ) {
			case FP64:    return new LiteralOp(Double.parseDouble(value));
			case INT64:   return new LiteralOp(Long.parseLong(value));
			case BOOLEAN: return new LiteralOp(Boolean.parseBoolean(value));
			case STRING:  return new LiteralOp(value);
			default: throw new RuntimeException("Unsupported scalar value type: "+vt.name());
		}
	}
	
	public static IntObject castToLong(ScalarObject so) {
		//note: cast with robustness for various combinations of value types
		return new IntObject(!(so instanceof StringObject) ?
			so.getLongValue() : UtilFunctions.toLong(Double.parseDouble(so.getStringValue())));
	}
	
	public static DoubleObject castToDouble(ScalarObject so) {
		//note: cast with robustness for various combinations of value types
		return new DoubleObject(!(so instanceof StringObject) ?
			so.getDoubleValue() : Double.parseDouble(so.getStringValue()));
	}
}
