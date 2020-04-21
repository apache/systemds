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

package org.apache.sysds.runtime.codegen;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.codehaus.janino.SimpleCompiler;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.CompilerType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;
import org.apache.sysds.runtime.codegen.SpoofOperator.SideInputSparseCell;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.utils.Statistics;

public class CodegenUtils 
{
	private static final Log LOG = LogFactory.getLog(CodegenUtils.class.getName());
	
	//cache to reuse compiled and loaded classes 
	private static ConcurrentHashMap<String, Class<?>> _cache = new ConcurrentHashMap<>();
	
	//janino-specific map of source code transfer/recompile on-demand
	private static ConcurrentHashMap<String, String> _src = new ConcurrentHashMap<>();
	
	//javac-specific working directory for src/class files
	private static String _workingDir = null;
	
	public static Class<?> compileClass(String name, String src) {
		//reuse existing compiled class
		Class<?> ret = _cache.get(name);
		if( ret != null ) 
			return ret;
		
		long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
		
		//compile java source w/ specific compiler
		if( SpoofCompiler.JAVA_COMPILER == CompilerType.JANINO )
			ret = compileClassJanino(name, src);
		else
			ret = compileClassJavac(name, src);
		
		//keep compiled class for reuse
		_cache.put(name, ret);
		
		if( DMLScript.STATISTICS ) {
			Statistics.incrementCodegenClassCompile();
			Statistics.incrementCodegenClassCompileTime(System.nanoTime()-t0);
		}
		
		return ret;
	}
	
	public static Class<?> getClass(String name) {
		return getClass(name, null);
	}
	
	public synchronized static Class<?> getClassSync(String name, byte[] classBytes) {
		//In order to avoid anomalies of concurrently compiling and loading the same
		//class with the same name multiple times in spark executors, this indirection
		//synchronizes the class compilation. This synchronization leads to the first
		//thread compiling the common class and all other threads simply reusing the
		//cached class instance, which also ensures that the same class is not loaded
		//multiple times which causes unnecessary JIT compilation overhead.
		return getClass(name, classBytes);
	}
	
	public static Class<?> getClass(String name, byte[] classBytes) {
		//reuse existing compiled class
		Class<?> ret = _cache.get(name);
		if( ret != null ) 
			return ret;
		
		//get class in a compiler-specific manner
		if( SpoofCompiler.JAVA_COMPILER == CompilerType.JANINO )
			ret = compileClassJanino(name, new String(classBytes));
		else
			ret = loadFromClassFile(name, classBytes);

		//keep loaded class for reuse
		_cache.put(name, ret);
		return ret;
	}
	
	public static byte[] getClassData(String name) {
		//get class in a compiler-specific manner
		if( SpoofCompiler.JAVA_COMPILER == CompilerType.JANINO )
			return _src.get(name).getBytes();
		else
			return getClassAsByteArray(name);
	}
	
	public static void clearClassCache() {
		_cache.clear();
		_src.clear();
	}
	
	public static void clearClassCache(Class<?> cla) {
		//one-pass, in-place filtering of class cache
		Iterator<Entry<String,Class<?>>> iter = _cache.entrySet().iterator();
		while( iter.hasNext() )
			if( iter.next().getValue()==cla )
				iter.remove();
	}
	
	public static SpoofOperator createInstance(Class<?> cla) {
		SpoofOperator ret = null;
		
		try {
			ret = (SpoofOperator) cla.newInstance();
		}
		catch( Exception ex ) {
			throw new DMLRuntimeException(ex);
		}
		
		return ret;
	}
	
	public static SideInput createSideInput(MatrixBlock in) {
		SideInput ret = (in.isInSparseFormat() || !in.isAllocated()) ?
			new SideInput(null, in, in.getNumColumns()) :
			new SideInput(in.getDenseBlock(), null, in.getNumColumns());
		return (ret.mdat != null) ? new SideInputSparseCell(ret) : ret;
	}
	
	////////////////////////////
	//JANINO-specific methods (used for spark environments)

	private static Class<?> compileClassJanino(String name, String src) {
		try {
			//compile source code
			SimpleCompiler compiler = new SimpleCompiler();
			compiler.cook(src);
			
			//keep source code for later re-construction
			_src.put(name, src);
			
			//load compile class
			return compiler.getClassLoader()
				.loadClass(name);
		}
		catch(Exception ex) {
			LOG.error("Failed to compile class "+name+": \n"+src);
			throw new DMLRuntimeException("Failed to compile class "+name+".", ex);
		}
	}	
	
	////////////////////////////
	//JAVAC-specific methods (used for hadoop environments)

	private static Class<?> compileClassJavac(String name, String src) {
		try
		{
			//create working dir on demand
			if( _workingDir == null )
				createWorkingDir();
			
			//write input file (for debugging / classpath handling)
			File ftmp = new File(_workingDir+"/"+name.replace(".", "/")+".java");
			if( !ftmp.getParentFile().exists() )
				ftmp.getParentFile().mkdirs();
			LocalFileUtils.writeTextFile(ftmp, src);
			
			//get system java compiler
			JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
			if( compiler == null )
				throw new RuntimeException("Unable to obtain system java compiler.");
		
			//prepare file manager
			DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>(); 
			try(StandardJavaFileManager fileManager = compiler.getStandardFileManager(diagnostics, null, null))
			{
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
				try (URLClassLoader classLoader = new URLClassLoader(
					new URL[]{new File(_workingDir).toURI().toURL(), runDir}, 
					CodegenUtils.class.getClassLoader()))
				{
					return classLoader.loadClass(name);
				}
			}
		}
		catch(Exception ex) {
			LOG.error("Failed to compile class "+name+": \n"+src);
			throw new DMLRuntimeException("Failed to compile class "+name+".", ex);
		}
	}
	
	private static Class<?> loadFromClassFile(String name, byte[] classBytes) {
		if(classBytes != null) {
			//load from byte representation of class file
			try(ByteClassLoader byteLoader = new ByteClassLoader(new URL[]{}, 
				CodegenUtils.class.getClassLoader(), classBytes)) 
			{
				return byteLoader.findClass(name);
			} 
			catch (Exception e) {
				throw new DMLRuntimeException(e);
			}
		}
		else {
			//load compiled class file
			URL runDir = CodegenUtils.class.getProtectionDomain().getCodeSource().getLocation(); 
			try(URLClassLoader classLoader = new URLClassLoader(new URL[]{new File(_workingDir)
				.toURI().toURL(), runDir}, CodegenUtils.class.getClassLoader())) 
			{
				return classLoader.loadClass(name);
			} 
			catch (Exception e) {
				throw new DMLRuntimeException(e);
			}
		}	
	}
	
	@SuppressWarnings("resource")
	private static byte[] getClassAsByteArray(String name) {
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
	
	private static void createWorkingDir() {
		if( _workingDir != null )
			return;
		String tmp = LocalFileUtils.getWorkingDir(LocalFileUtils.CATEGORY_CODEGEN);
		LocalFileUtils.createLocalFileIfNotExist(tmp);
		_workingDir = tmp;
	}
}
