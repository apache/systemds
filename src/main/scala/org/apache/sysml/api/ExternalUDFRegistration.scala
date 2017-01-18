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

import scala.reflect.runtime.universe._
import java.util.ArrayList
import org.apache.sysml.udf.FunctionParameter
import org.apache.sysml.udf.Scalar
import org.apache.sysml.udf.Matrix
import org.apache.sysml.udf.Matrix.ValueType
import org.apache.sysml.api.mlcontext.Script
import org.apache.sysml.udf.PackageFunction
import org.apache.sysml.udf.FunctionParameter
import org.apache.sysml.udf.lib.GenericFunction
import org.apache.sysml.udf.Scalar.ScalarValueType
import java.util.HashMap

object ExternalUDFRegistration {
  val fnMapping: HashMap[String, Function0[Array[FunctionParameter]]] = new HashMap[String, Function0[Array[FunctionParameter]]]()
  val fnSignatureMapping: HashMap[String, Array[String]] = new HashMap[String, Array[String]]()
  val udfMapping:HashMap[String, GenericFunction] = new HashMap[String, GenericFunction]();
}

/**
 * This class handles the registration of external Scala UDFs via MLContext.
 */
class ExternalUDFRegistration {
  var ml:MLContext = null
  def setMLContext(ml1:org.apache.sysml.api.mlcontext.MLContext) = { this.ml = ml }
  
  val scriptHeader:StringBuilder = new StringBuilder
  def addHeaders(script:Script): Unit = {
    val header = scriptHeader.toString() 
    if(!header.equals("")) {
			script.setScriptString(scriptHeader + "\n" + script.getScriptString());
			System.out.println(script.getScriptString)
    }
  }
  
  def getType(t: String):String = {
    t match {
      case "java.lang.String" => "string"
      case "Double" => "double"
      case "Int" => "integer"
      case "Boolean" => "boolean"
      // Support only pass by value for now.
      // case "org.apache.sysml.runtime.matrix.data.MatrixBlock" => "matrix[double]"
      // case "scala.Array[Double]" => "matrix[double]"
      case "scala.Array[scala.Array[Double]]" => "matrix[double]"
      case _ => throw new RuntimeException("Unsupported type of parameter: " + t)
    }
  }
   
   // zero-input function unsupported by SystemML
//  def register[RT: TypeTag](name: String, func: Function0[RT]): Unit = {
//    println(getType(typeOf[RT].toString()))
//  }
   
   def register[A1: TypeTag, RT: TypeTag](name: String, func: Function1[A1, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(typeOf[A1].toString(), typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    scriptHeader.append(name + " = externalFunction(")
    scriptHeader.append(getType(typeOf[A1].toString()) + " input1")
    scriptHeader.append(") return (")
    // TODO: Support multiple return types
    scriptHeader.append(getType(typeOf[RT].toString())  + " output1")
    scriptHeader.append(") implemented in (classname=\"org.apache.sysml.udf.lib.GenericFunction\", exectype=\"mem\");\n")
  }
  
  def convertReturnToOutput(ret:Any): Array[FunctionParameter] = {
    ret match {
       case x:Tuple1[Any] => Array(convertToOutput(x._1))
       case x:Tuple2[Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2))
//       case x:Tuple3[B1, B2, B3] => Array(convertToOutput(x._1), convertToOutput(x._2))
//       case x:Tuple4[B1, B2, B3, B4] => Array(convertToOutput(x._1), convertToOutput(x._2))
//       case x:Tuple5[B1, B2, B3, B4, B5] => Array(convertToOutput(x._1), convertToOutput(x._2))
       case _ => Array(convertToOutput(ret))
     }
  }
   val rand = new java.util.Random()
   def convertToOutput(x:Any): FunctionParameter = {
     x match {
       case x1:Int => return new Scalar(ScalarValueType.Integer, String.valueOf(x))
       case x1:java.lang.Integer => return new Scalar(ScalarValueType.Integer, String.valueOf(x))
       case x1:Double => return new Scalar(ScalarValueType.Double, String.valueOf(x))
       case x1:java.lang.Double => return new Scalar(ScalarValueType.Double, String.valueOf(x))
       case x1:java.lang.String => return new Scalar(ScalarValueType.Text, String.valueOf(x))
       case x1:java.lang.Boolean => return new Scalar(ScalarValueType.Boolean, String.valueOf(x))
       case x1:Boolean => return new Scalar(ScalarValueType.Boolean, String.valueOf(x))
       case x1:scala.Array[scala.Array[Double]] => {
         val mat = new Matrix( "temp" + rand.nextLong, x1.length, x1(0).length, ValueType.Double );
			   mat.setMatrixDoubleArray(x1)
			   return mat
       }
       case _ => throw new RuntimeException("Unsupported output type:" + x.getClass().getName)
     }
   }
}