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

package org.apache.sysml.runtime.codegen;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import javax.tools.Diagnostic;
import javax.tools.Diagnostic.Kind;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaCompiler.CompilationTask;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

import org.apache.commons.io.IOUtils;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.util.LocalFileUtils;
import org.apache.sysml.utils.Statistics;

public class CodegenUtils 
{
	//cache to reuse compiled and loaded classes 
	private static ConcurrentHashMap<String, Class<?>> _cache = new ConcurrentHashMap<String,Class<?>>();
	private static String _workingDir = null;
	
	public static Class<?> compileClass(String name, String src) 
		throws DMLRuntimeException
	{
		//reuse existing compiled class
		Class<?> ret = _cache.get(name);
		if( ret != null ) 
			return ret;
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		try
		{
			//create working dir on demand
			if( _workingDir == null )
				createWorkingDir();
			
			//write input file (for debugging / classpath handling)
			File ftmp = new File(_workingDir+"/codegen/"+name+".java");
			if( !ftmp.getParentFile().exists() )
				ftmp.getParentFile().mkdirs();
			LocalFileUtils.writeTextFile(ftmp, src);
			
			//get system java compiler
			JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
			if( compiler == null )
				throw new RuntimeException("Unable to obtain system java compiler.");
		
			//prepare file manager
			DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<JavaFileObject>(); 
			StandardJavaFileManager fileManager = compiler.getStandardFileManager(diagnostics, null, null);
			
			//prepare input source code
			Iterable<? extends JavaFileObject> sources = fileManager
					.getJavaFileObjectsFromFiles(Arrays.asList(ftmp));
			
			//prepare class path 
			URL runDir = CodegenUtils.class.getProtectionDomain().getCodeSource().getLocation(); 
			String classpath = System.getProperty("java.class.path") + 
					File.pathSeparator + runDir.getPath();
			List<String> options = Arrays.asList("-classpath",classpath);
			
			//compile source code
			CompilationTask task = compiler.getTask(null, fileManager, diagnostics, options, null, sources);
			Boolean success = task.call();
			
			//output diagnostics and error handling
			for(Diagnostic<? extends JavaFileObject> tmp : diagnostics.getDiagnostics())
				if( tmp.getKind()==Kind.ERROR )
					System.err.println("ERROR: "+tmp.toString());				
			if( success == null || !success )
				throw new RuntimeException("Failed to compile class "+name);
			
			//dynamically load compiled class
			URLClassLoader classLoader = null;
			try {
				classLoader = new URLClassLoader(
					new URL[]{new File(_workingDir).toURI().toURL(), runDir}, 
					CodegenUtils.class.getClassLoader());
				ret = classLoader.loadClass("codegen."+name);
			}
			finally {
				IOUtilFunctions.closeSilently(classLoader);
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		//keep compiled class for reuse
		_cache.put(name, ret);
		
		if( DMLScript.STATISTICS ) {
			Statistics.incrementCodegenClassCompile();
			Statistics.incrementCodegenClassCompileTime(System.nanoTime()-t0);
		}
		
		return ret;
	}
	
	public static Class<?> loadClass(String name) throws DMLRuntimeException {
		return loadClass(name, null);
	}
	
	public static Class<?> loadClass(String name, byte[] classBytes) 
		throws DMLRuntimeException 
	{
		//reuse existing compiled class
		Class<?> ret = _cache.get(name);
		if( ret != null ) 
			return ret;
		
		//define class using the bytes
		if(classBytes != null)
		{
			//ByteClassLoader byteLoader = new ByteClassLoader(classLoader.getURLs() , classLoader.getParent(), classBytes);
			try {
				ByteClassLoader byteLoader = new ByteClassLoader(new URL[]{} ,CodegenUtils.class.getClassLoader(), classBytes);
				ret = byteLoader.findClass(name);
				byteLoader.close();
			} catch (Exception e) {
				throw new DMLRuntimeException(e);
			}
		}
		else
		{
			//dynamically load compiled class
			URL runDir = CodegenUtils.class.getProtectionDomain().getCodeSource().getLocation(); 
			URLClassLoader classLoader = null;
			try {
				classLoader = new URLClassLoader(
						new URL[]{new File(_workingDir).toURI().toURL(), runDir}, 
						CodegenUtils.class.getClassLoader());
				ret = classLoader.loadClass(name);
			} 
			catch (Exception e) {
				throw new DMLRuntimeException(e);
			}
			finally {
				IOUtilFunctions.closeSilently(classLoader);
			}
		}
		
		//keep loaded class for reuse
		_cache.put(name, ret);
		return ret;
	}
	
	public static Object createInstance(Class<?> cla) 
		throws DMLRuntimeException 
	{
		Object ret = null;
		
		try {
			ret = cla.newInstance();	
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException(ex);
		}
		
		return ret;
	}
	
	public static byte[] getClassAsByteArray(String name) 
		throws DMLRuntimeException
	{
		String classAsPath = name.replace('.', '/') + ".class";
		
		URLClassLoader classLoader = null;
		InputStream stream = null;
		
		try {
			//dynamically load compiled class
			URL runDir = CodegenUtils.class.getProtectionDomain().getCodeSource().getLocation(); 
			classLoader = new URLClassLoader(
					new URL[]{new File(_workingDir).toURI().toURL(), runDir}, 
					CodegenUtils.class.getClassLoader());
			stream = classLoader.getResourceAsStream(classAsPath);
			return IOUtils.toByteArray(stream);
		} 
		catch (IOException e) {
			throw new DMLRuntimeException(e);
		}
		finally {
			IOUtilFunctions.closeSilently(classLoader);
			IOUtilFunctions.closeSilently(stream);
		}
	}

	public static String getSpoofType(Class<?> cls) {
		if(cls.getSuperclass() == SpoofCellwise.class)
			return "Cell" +  cls.getName().split("\\.")[1];
		else if(cls.getSuperclass() == SpoofOuterProduct.class)
			return "OP" +  cls.getName().split("\\.")[1];
		else if(cls.getSuperclass() == SpoofRowAggregate.class)
			return "RA" +  cls.getName().split("\\.")[1];
		else
			return "UNKNOWN";
	}
	
	public static void clearClassCache() {
		_cache.clear();
	}
	
	private static void createWorkingDir() throws DMLRuntimeException  {
		if( _workingDir != null )
			return;
		String tmp = LocalFileUtils.getWorkingDir(LocalFileUtils.CATEGORY_CODEGEN);
		LocalFileUtils.createLocalFileIfNotExist(tmp);
		_workingDir = tmp;
	}
}
