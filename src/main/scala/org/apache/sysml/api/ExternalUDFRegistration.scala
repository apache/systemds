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

/*
 * Design of Scala external UDF functionality:
 * Two main classes in that enable this functionality are as follows:
 * 1. ExternalUDFRegistration: We have overloaded the register method to allow for registration
 * of scala UDFs with 10 arguments. Each of these functions examine the input types to check
 * if they are supported (see getType). If input types are supported, then it creates a header of format:
 * 
 * fnName = externalFunction(input arguments) return (output arguments) implemented in (classname="org.apache.sysml.udf.lib.GenericFunction",exectype="mem")
 * 
 * This header is appended in MLContext before execution of the script.
 * 
 * In addition, it populates two global data structures: fnMapping (which stores a zero-argument anonymous
 * function) and fnSignatureMapping (useful for computing the number of return values).
 * These data structures are used by GenericFunction.
 * 
 * The secret sauce of this approach is conversion of arbitrary Scala UDF into a zero-argument anonymous UDF
 * stored in ExternalUDFRegistration's fnMapping data structure (similar to execute) :)
 * 
 * 2. GenericFunction
 * This generic class is called by SystemML for any registered Scala UDF. This class first inserts itself into
 * ExternalUDFRegistration's udfMapping data structure and then invokes the zero-argument anonymous
 * function corresponding to the user specified Scala UDF.
 *  
 * 
 * The current implementation allows the functions registered with one MLContext 
 * to be visible to other MLContext as well as ExternalUDFRegistration's fnMapping, fnSignatureMapping and udfMapping
 * fields are static. This is necessary to simplify the integration with existing external UDF function framework.
 * 
 * Usage:
 * scala> import org.apache.sysml.api.mlcontext._
 * scala> import org.apache.sysml.api.mlcontext.ScriptFactory._
 * scala> val ml = new MLContext(sc)
 * scala> 
 * scala> // Demonstrates how to pass a simple scala UDF to SystemML
 * scala> def addOne(x:Double):Double = x + 1
 * scala> ml.udf.register("addOne", addOne)
 * scala> val script1 = dml("v = addOne(2.0); print(v)")
 * scala> ml.execute(script1)
 * scala> 
 * scala> // Demonstrates operation on local matrices (double[][])
 * scala> def addOneToDiagonal(x:Array[Array[Double]]):Array[Array[Double]] = {  for(i <- 0 to x.length-1) x(i)(i) = x(i)(i) + 1; x }
 * scala> ml.udf.register("addOneToDiagonal", addOneToDiagonal)
 * scala> val script2 = dml("m1 = matrix(0, rows=3, cols=3); m2 = addOneToDiagonal(m1); print(toString(m2));")
 * scala> ml.execute(script2)
 * scala> 
 * scala> // Demonstrates multi-return function
 * scala> def multiReturnFn(x:Double):(Double, Int) = (x + 1, (x * 2).toInt)
 * scala> ml.udf.register("multiReturnFn", multiReturnFn)
 * scala> val script3 = dml("[v1, v2] = multiReturnFn(2.0); print(v1)")
 * scala> ml.execute(script3)
 * scala> 
 * scala> // Demonstrates multi-argument multi-return function
 * scala> def multiArgReturnFn(x:Double, y:Int):(Double, Int) = (x + 1, (x * y).toInt)
 * scala> ml.udf.register("multiArgReturnFn", multiArgReturnFn _)
 * scala> val script4 = dml("[v1, v2] = multiArgReturnFn(2.0, 1); print(v2)")
 * scala> ml.execute(script4)
 */

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
			// Useful for debugging:
			// System.out.println(script.getScriptString)
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
  
  def getReturnType(t: String):String = {
    if(t.startsWith("(")) {
      val t1 = t.substring(1, t.length()-1).split(",").map(_.trim)
      val ret = new StringBuilder
      for(i <- 0 until t1.length) {
        if(i != 0) ret.append(", ")
        ret.append(getType(t1(i)) + " output" + i)
      }
      ret.toString
    }
    else
      getType(t) + " output0"
  }
  
  def appendHead(name:String): Unit = {
    scriptHeader.append(name + " = externalFunction(")
  }
  def appendTail(typeRet:String): Unit = {
    scriptHeader.append(") return (")
    scriptHeader.append(getReturnType(typeRet))
    scriptHeader.append(") implemented in (classname=\"org.apache.sysml.udf.lib.GenericFunction\", exectype=\"mem\");\n")
  }
  
  // ------------------------------------------------------------------------------------------
  // Overloaded register function for 1 to 10 inputs:
  
   // zero-input function unsupported by SystemML
//  def register[RT: TypeTag](name: String, func: Function0[RT]): Unit = {
//    println(getType(typeOf[RT].toString()))
//  }
   
  def unregister(name: String): Unit = {
    ExternalUDFRegistration.fnSignatureMapping.remove(name)
    ExternalUDFRegistration.fnMapping.remove(name)
    ExternalUDFRegistration.udfMapping.remove(name)
  }
  
   def register[A1: TypeTag, RT: TypeTag](name: String, func: Function1[A1, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(typeOf[A1].toString(), typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, RT: TypeTag](name: String, func: Function2[A1, A2, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(typeOf[A1].toString(), typeOf[A2].toString(), typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, RT: TypeTag](name: String, func: Function3[A1, A2, A3, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), 
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, RT: TypeTag](name: String, func: Function4[A1, A2, A3, A4, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(), 
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, A5: TypeTag, RT: TypeTag](name: String, 
      func: Function5[A1, A2, A3, A4, A5, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4], 
             udf.getInput(typeOf[A5].toString(), 4).asInstanceOf[A5]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(),
        typeOf[A5].toString(),
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3, ")
    scriptHeader.append(getType(typeOf[A5].toString()) + " input4")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, A5: TypeTag, A6: TypeTag, RT: TypeTag](name: String, 
      func: Function6[A1, A2, A3, A4, A5, A6, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4], 
             udf.getInput(typeOf[A5].toString(), 4).asInstanceOf[A5], 
             udf.getInput(typeOf[A6].toString(), 5).asInstanceOf[A6]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(),
        typeOf[A5].toString(), typeOf[A6].toString(),
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3, ")
    scriptHeader.append(getType(typeOf[A5].toString()) + " input4, ")
    scriptHeader.append(getType(typeOf[A6].toString()) + " input5")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, A5: TypeTag, A6: TypeTag, A7: TypeTag, RT: TypeTag](name: String, 
      func: Function7[A1, A2, A3, A4, A5, A6, A7, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4], 
             udf.getInput(typeOf[A5].toString(), 4).asInstanceOf[A5], 
             udf.getInput(typeOf[A6].toString(), 5).asInstanceOf[A6],
             udf.getInput(typeOf[A7].toString(), 6).asInstanceOf[A7]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(),
        typeOf[A5].toString(), typeOf[A6].toString(), typeOf[A7].toString(),
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3, ")
    scriptHeader.append(getType(typeOf[A5].toString()) + " input4, ")
    scriptHeader.append(getType(typeOf[A6].toString()) + " input5, ")
    scriptHeader.append(getType(typeOf[A7].toString()) + " input6")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, A5: TypeTag, A6: TypeTag, A7: TypeTag, 
    A8: TypeTag, RT: TypeTag](name: String, 
      func: Function8[A1, A2, A3, A4, A5, A6, A7, A8, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4], 
             udf.getInput(typeOf[A5].toString(), 4).asInstanceOf[A5], 
             udf.getInput(typeOf[A6].toString(), 5).asInstanceOf[A6],
             udf.getInput(typeOf[A7].toString(), 6).asInstanceOf[A7], 
             udf.getInput(typeOf[A8].toString(), 7).asInstanceOf[A8]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(),
        typeOf[A5].toString(), typeOf[A6].toString(), typeOf[A7].toString(), typeOf[A8].toString(),
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3, ")
    scriptHeader.append(getType(typeOf[A5].toString()) + " input4, ")
    scriptHeader.append(getType(typeOf[A6].toString()) + " input5, ")
    scriptHeader.append(getType(typeOf[A7].toString()) + " input6, ")
    scriptHeader.append(getType(typeOf[A8].toString()) + " input7")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, A5: TypeTag, A6: TypeTag, A7: TypeTag, 
    A8: TypeTag, A9: TypeTag, RT: TypeTag](name: String, 
      func: Function9[A1, A2, A3, A4, A5, A6, A7, A8, A9, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4], 
             udf.getInput(typeOf[A5].toString(), 4).asInstanceOf[A5], 
             udf.getInput(typeOf[A6].toString(), 5).asInstanceOf[A6],
             udf.getInput(typeOf[A7].toString(), 6).asInstanceOf[A7], 
             udf.getInput(typeOf[A8].toString(), 7).asInstanceOf[A8], 
             udf.getInput(typeOf[A9].toString(), 8).asInstanceOf[A9]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(),
        typeOf[A5].toString(), typeOf[A6].toString(), typeOf[A7].toString(), typeOf[A8].toString(),
        typeOf[A9].toString(),
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3, ")
    scriptHeader.append(getType(typeOf[A5].toString()) + " input4, ")
    scriptHeader.append(getType(typeOf[A6].toString()) + " input5, ")
    scriptHeader.append(getType(typeOf[A7].toString()) + " input6, ")
    scriptHeader.append(getType(typeOf[A8].toString()) + " input7, ")
    scriptHeader.append(getType(typeOf[A9].toString()) + " input8")
    appendTail(typeOf[RT].toString())
  }
  
  def register[A1: TypeTag, A2: TypeTag, A3: TypeTag, A4: TypeTag, A5: TypeTag, A6: TypeTag, A7: TypeTag, 
    A8: TypeTag, A9: TypeTag, A10: TypeTag, RT: TypeTag](name: String, 
      func: Function10[A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, RT]): Unit = {
    val anonfun0 = new Function0[Array[FunctionParameter]] {
       def apply(): Array[FunctionParameter] = {
         val udf = ExternalUDFRegistration.udfMapping.get(name);
         return convertReturnToOutput(func.apply(udf.getInput(typeOf[A1].toString(), 0).asInstanceOf[A1],
             udf.getInput(typeOf[A2].toString(), 1).asInstanceOf[A2], 
             udf.getInput(typeOf[A3].toString(), 2).asInstanceOf[A3],
             udf.getInput(typeOf[A4].toString(), 3).asInstanceOf[A4], 
             udf.getInput(typeOf[A5].toString(), 4).asInstanceOf[A5], 
             udf.getInput(typeOf[A6].toString(), 5).asInstanceOf[A6],
             udf.getInput(typeOf[A7].toString(), 6).asInstanceOf[A7], 
             udf.getInput(typeOf[A8].toString(), 7).asInstanceOf[A8], 
             udf.getInput(typeOf[A9].toString(), 8).asInstanceOf[A9],
             udf.getInput(typeOf[A10].toString(), 9).asInstanceOf[A10]))
       }
    }
    ExternalUDFRegistration.fnSignatureMapping.put(name, Array(
        typeOf[A1].toString(), typeOf[A2].toString(), typeOf[A3].toString(), typeOf[A4].toString(),
        typeOf[A5].toString(), typeOf[A6].toString(), typeOf[A7].toString(), typeOf[A8].toString(),
        typeOf[A9].toString(), typeOf[A10].toString(),
        typeOf[RT].toString()))
    ExternalUDFRegistration.fnMapping.put(name, anonfun0);
    appendHead(name)
    scriptHeader.append(getType(typeOf[A1].toString()) + " input0, ")
    scriptHeader.append(getType(typeOf[A2].toString()) + " input1, ")
    scriptHeader.append(getType(typeOf[A3].toString()) + " input2, ")
    scriptHeader.append(getType(typeOf[A4].toString()) + " input3, ")
    scriptHeader.append(getType(typeOf[A5].toString()) + " input4, ")
    scriptHeader.append(getType(typeOf[A6].toString()) + " input5, ")
    scriptHeader.append(getType(typeOf[A7].toString()) + " input6, ")
    scriptHeader.append(getType(typeOf[A8].toString()) + " input7, ")
    scriptHeader.append(getType(typeOf[A9].toString()) + " input8, ")
    scriptHeader.append(getType(typeOf[A10].toString()) + " input9")
    appendTail(typeOf[RT].toString())
  }
  
  // ------------------------------------------------------------------------------------------
  
  def convertReturnToOutput(ret:Any): Array[FunctionParameter] = {
    ret match {
       case x:Tuple1[Any] => Array(convertToOutput(x._1))
       case x:Tuple2[Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2))
       case x:Tuple3[Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3))
       case x:Tuple4[Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4))
       case x:Tuple5[Any, Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4), convertToOutput(x._5))
       case x:Tuple6[Any, Any, Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4), convertToOutput(x._5), convertToOutput(x._6))
       case x:Tuple7[Any, Any, Any, Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4), convertToOutput(x._5), convertToOutput(x._6), convertToOutput(x._7))
       case x:Tuple8[Any, Any, Any, Any, Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4), convertToOutput(x._5), convertToOutput(x._6), convertToOutput(x._7), convertToOutput(x._8))
       case x:Tuple9[Any, Any, Any, Any, Any, Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4), convertToOutput(x._5), convertToOutput(x._6), convertToOutput(x._7), 
                                                                 convertToOutput(x._8), convertToOutput(x._9))
       case x:Tuple10[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any] => Array(convertToOutput(x._1), convertToOutput(x._2), convertToOutput(x._3), convertToOutput(x._4), convertToOutput(x._5), convertToOutput(x._6), convertToOutput(x._7), 
                                                                 convertToOutput(x._8), convertToOutput(x._9), convertToOutput(x._10))                                                          
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