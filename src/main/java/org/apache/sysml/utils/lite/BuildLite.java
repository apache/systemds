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
package org.apache.sysml.utils.lite;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Set;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.Vector;
import java.util.jar.Attributes;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.jar.Manifest;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.compress.archivers.jar.JarArchiveEntry;
import org.apache.commons.compress.archivers.jar.JarArchiveInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.random.Well1024a;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.ThrowableInformation;

/**
 * Builds a light-weight SystemML jar file based on loaded classes and
 * additional resources. Additionally generates maven assembly dependency sets
 * that are used by the lite.xml assembly. Note that the jar file automatically
 * built by createLiteJar will only contain required SystemML classes, whereas
 * the assembly jar file (built by lite.xml) includes all SystemML classes. All
 * log4j classes are included in both the automatic jar and the assembly jar.
 * All commons-math3 classes are included by default in both the automatic jar
 * and the assembly jar, but this can be switched using createLiteJar to only
 * include the detected required commons-math3 classes.
 *
 */
public class BuildLite {

	/**
	 * Default lite jar path and name.
	 */
	public static final String DEFAULT_LITE_JAR_LOCATION = "systemml-lite.jar";

	/**
	 * File within the lite jar that can be used to identify execution from the
	 * lite jar.
	 */
	public static final String LITE_JAR_IDENTIFIER_FILE = "META-INF/systemml-lite.txt";

	/**
	 * The lite jar path and name.
	 */
	private static String liteJarLocation = DEFAULT_LITE_JAR_LOCATION;

	/**
	 * Additional resources that should be added to the lite jar file. This can
	 * include resources such as service files and shutdown hooks that aren't
	 * detected by query to the classloader.
	 */
	public static List<String> additionalResources = new ArrayList<>();
	static {
		// avoid "No FileSystem for scheme: file" error in JMLC
		additionalResources.add("META-INF/services/org.apache.hadoop.fs.FileSystem");
		// shutdown hook class
		additionalResources.add("org/apache/hadoop/util/ShutdownHookManager$2.class");

		additionalResources.add("org/apache/hadoop/log/metrics/EventCounter.class");
	}

	/**
	 * Map jars to the additional resources files in order to build the
	 * dependency sets required by lite.xml.
	 */
	public static SortedMap<String, SortedSet<String>> additionalJarToFileMappingsForDependencySets = new TreeMap<>();
	static {
		SortedSet<String> hadoopCommonResources = new TreeSet<>();
		hadoopCommonResources.add("META-INF/services/org.apache.hadoop.fs.FileSystem");
		hadoopCommonResources.add("org/apache/hadoop/util/ShutdownHookManager$2.class");
		hadoopCommonResources.add("org/apache/hadoop/log/metrics/EventCounter.class");
		additionalJarToFileMappingsForDependencySets.put("hadoop-common", hadoopCommonResources);
	}

	/**
	 * Scan project *.java files for these packages/classes that should
	 * definitely be included in the lite jar.
	 */
	public static List<String> additionalPackages = new ArrayList<>();
	static {
		// math3, lang3, io, etc.
		additionalPackages.add("org.apache.commons");
	}

	/**
	 * Exclude classes of the following packages from the lite jar.
	 */
	public static List<String> packagesToExclude = new ArrayList<>();
	static {
		packagesToExclude.add("com.sun.proxy");
		// these can be added if test suite code is run
		packagesToExclude.add("org.junit");
		packagesToExclude.add("org.apache.spark");
		packagesToExclude.add("org.apache.sysml.test");
		packagesToExclude.add("scala");
	}

	/**
	 * The base source directory to scan for classes to potentially load.
	 */
	private static final String BASE_SRC_DIR = "src/main";

	/**
	 * The detected classes and their jar files, where the jar names are keys in
	 * a sorted map and sorted sets of the class names are the values in the
	 * sorted map.
	 */
	public static SortedMap<String, SortedSet<String>> jarsAndClasses = new TreeMap<>();

	/**
	 * The jar dependencies and their sizes for comparison purposes.
	 */
	public static SortedMap<String, Long> jarSizes = new TreeMap<>();

	/**
	 * Dummy logger to fill in log4j info for things such as the jar sizes.
	 * Usually not needed.
	 */
	protected static Logger log = Logger.getLogger(BuildLite.class);

	private static boolean includeAllCommonsMath3 = true;

	/**
	 * Create lite jar file using the default path and file name as the
	 * destination. All commons-math3 classes will be included.
	 * 
	 * @throws Exception
	 *             if exception occurs building jar
	 */
	public static void createLiteJar() throws Exception {
		createLiteJar(null, true);
	}

	/**
	 * Create lite jar file using the default path and file name as the
	 * destination, specifying whether all commons-math3 classes should be
	 * included in the jar or only the detected required subset.
	 * 
	 * @param allCommonsMath3
	 *            if true, include all commons-math3 classes. if false, include
	 *            only required subset in jar built
	 * @throws Exception
	 *             if exception occurs building jar
	 */
	public static void createLiteJar(boolean allCommonsMath3) throws Exception {
		createLiteJar(null, allCommonsMath3);
	}

	/**
	 * Create lite jar file specifying the destination path and file name as a
	 * string.
	 * 
	 * @param jarFileDestination
	 *            the path and file name for the lite jar
	 * @param allCommonsMath3
	 *            if true, include all commons-math3 classes. if false, include
	 *            only required subset in jar built
	 * @throws Exception
	 *             if exception occurs building jar
	 */
	public static void createLiteJar(String jarFileDestination, boolean allCommonsMath3) throws Exception {
		if (jarFileDestination != null) {
			liteJarLocation = jarFileDestination;
		}
		includeAllCommonsMath3 = allCommonsMath3;
		scanJavaFilesForClassesToLoad();
		List<Class<?>> loadedClasses = getLoadedClasses();
		displayLoadedClasses(loadedClasses);
		excludePackages(loadedClasses);
		displayLoadedClasses(loadedClasses);
		groupLoadedClassesByJarAndClass(loadedClasses);
		List<String> log4jClassPathNames = getLog4jClassPathNames();
		displayLog4JClassPathNames(log4jClassPathNames);
		List<String> commonsMath3ClassPathNames = null;
		if (includeAllCommonsMath3) {
			commonsMath3ClassPathNames = getCommonsMath3ClassPathNames();
			displayCommonsMath3ClassPathNames(commonsMath3ClassPathNames);
		}
		displayJarsAndClasses();
		Set<String> consolidatedClassPathNames = consolidateClassPathNames(loadedClasses, log4jClassPathNames,
				commonsMath3ClassPathNames);
		createJarFromConsolidatedClassPathNames(consolidatedClassPathNames);
		createDependencySets();
		displayJarSizes();
		liteJarStats();
	}

	/**
	 * Exclude selected packages from the loaded classes.
	 * 
	 * @param loadedClasses
	 *            classes that have been loaded by the classloader
	 */
	private static void excludePackages(List<Class<?>> loadedClasses) {
		System.out.println("\nExcluding selected packages");
		int count = 0;
		for (int i = 0; i < loadedClasses.size(); i++) {
			Class<?> clazz = loadedClasses.get(i);
			String className = clazz.getName();
			for (String packageToExclude : packagesToExclude) {
				if (className.startsWith(packageToExclude)) {
					System.out.println(" #" + ++count + ": Excluding " + className);
					loadedClasses.remove(i);
					i--;
				}
			}
		}

	}

	/**
	 * Statistics about the lite jar such as jar size.
	 */
	private static void liteJarStats() {
		File f = new File(liteJarLocation);
		if (f.exists()) {
			Long jarSize = f.length();
			String jarSizeDisplay = FileUtils.byteCountToDisplaySize(jarSize);
			System.out.println("\nFinished creating " + liteJarLocation + " (" + jarSizeDisplay + " ["
					+ NumberFormat.getInstance().format(jarSize) + " bytes])");
		} else {
			System.out.println(liteJarLocation + " could not be found");
		}
	}

	/**
	 * Consolidate the loaded classes and all the log4j classes and potentially
	 * all the commons-math3 classes.
	 * 
	 * @param loadedClasses
	 *            the loaded classes
	 * @param log4jClassPathNames
	 *            the log4j class names
	 * @param commonsMath3ClassPathNames
	 *            the commons-math3 class names
	 * @return the set of unique class names that combines the loaded classes
	 *         and the log4j classes
	 */
	private static Set<String> consolidateClassPathNames(List<Class<?>> loadedClasses, List<String> log4jClassPathNames,
			List<String> commonsMath3ClassPathNames) {

		SortedSet<String> allClassPathNames = new TreeSet<>(log4jClassPathNames);
		if (includeAllCommonsMath3) {
			System.out.println("\nConsolidating loaded class names, log4j class names, and commons-math3 class names");
			allClassPathNames.addAll(commonsMath3ClassPathNames);
		} else {
			System.out.println("\nConsolidating loaded class names and log4j class names");
		}
		for (Class<?> clazz : loadedClasses) {
			String loadedClassPathName = clazz.getName();
			loadedClassPathName = loadedClassPathName.replace(".", "/");
			loadedClassPathName = loadedClassPathName + ".class";
			allClassPathNames.add(loadedClassPathName);
		}
		return allClassPathNames;
	}

	/**
	 * Build a lite jar based on the consolidated class names.
	 * 
	 * @param consolidateClassPathNames
	 *            the consolidated class names
	 * @throws IOException
	 *             if an IOException occurs
	 */
	private static void createJarFromConsolidatedClassPathNames(Set<String> consolidateClassPathNames)
			throws IOException {
		System.out.println("\nCreating " + liteJarLocation + " file");
		ClassLoader cl = BuildLite.class.getClassLoader();

		Manifest mf = new Manifest();
		Attributes attr = mf.getMainAttributes();
		attr.putValue("" + Attributes.Name.MANIFEST_VERSION, "1.0");

		File file = new File(liteJarLocation);
		try (FileOutputStream fos = new FileOutputStream(file); JarOutputStream jos = new JarOutputStream(fos, mf)) {
			int numFilesWritten = 0;
			for (String classPathName : consolidateClassPathNames) {
				writeMessage(classPathName, ++numFilesWritten);
				InputStream is = cl.getResourceAsStream(classPathName);
				byte[] bytes = IOUtils.toByteArray(is);

				JarEntry je = new JarEntry(classPathName);
				jos.putNextEntry(je);
				jos.write(bytes);
			}

			writeIdentifierFileToLiteJar(jos, ++numFilesWritten);
			writeAdditionalResourcesToJar(jos, numFilesWritten);
		}

	}

	/**
	 * Write an identifier file to the lite jar that can be used to identify
	 * that the lite jar is being used.
	 * 
	 * @param jos
	 *            output stream to the jar being written
	 * @param numFilesWritten
	 *            the number of files written to the jar so far
	 * @throws IOException
	 *             if an IOException occurs
	 */
	private static void writeIdentifierFileToLiteJar(JarOutputStream jos, int numFilesWritten) throws IOException {
		writeMessage(LITE_JAR_IDENTIFIER_FILE, numFilesWritten);
		JarEntry je = new JarEntry(LITE_JAR_IDENTIFIER_FILE);
		jos.putNextEntry(je);
		String created = "Created " + (new Date());
		String userName = System.getProperty("user.name");
		if (userName != null) {
			created = created + " by " + userName;
		}
		jos.write(created.getBytes());
	}

	/**
	 * Write the additional resources to the jar.
	 * 
	 * @param jos
	 *            output stream to the jar being written
	 * @param numFilesWritten
	 *            the number of files written to the jar so far
	 * @throws IOException
	 *             if an IOException occurs
	 */
	private static void writeAdditionalResourcesToJar(JarOutputStream jos, int numFilesWritten) throws IOException {
		for (String resource : additionalResources) {
			writeMessage(resource, ++numFilesWritten);
			JarEntry je = new JarEntry(resource);
			jos.putNextEntry(je);
			ClassLoader cl = BuildLite.class.getClassLoader();
			InputStream is = cl.getResourceAsStream(resource);
			byte[] bytes = IOUtils.toByteArray(is);
			jos.write(bytes);
		}
	}

	/**
	 * Output message about the resource being written to the lite jar.
	 * 
	 * @param resource
	 *            the path to the resource
	 * @param numFilesWritten
	 *            the number of files written to the jar so far
	 */
	private static void writeMessage(String resource, int numFilesWritten) {
		System.out.println(" #" + numFilesWritten + ": Writing " + resource + " to " + liteJarLocation);
	}

	/**
	 * Obtain a list of all log4j classes in the referenced log4j jar file.
	 * 
	 * @return list of all the log4j classes in the referenced log4j jar file
	 * @throws IOException
	 *             if an IOException occurs
	 * @throws ClassNotFoundException
	 *             if a ClassNotFoundException occurs
	 */
	private static List<String> getLog4jClassPathNames() throws IOException, ClassNotFoundException {
		return getAllClassesInJar(ThrowableInformation.class);
	}

	/**
	 * Obtain a list of all commons-math3 classes in the referenced
	 * commons-math3 jar file.
	 * 
	 * @return list of all the commons-math3 classes in the referenced
	 *         commons-math3 jar file
	 * @throws IOException
	 *             if an IOException occurs
	 * @throws ClassNotFoundException
	 *             if a ClassNotFoundException occurs
	 */
	private static List<String> getCommonsMath3ClassPathNames() throws IOException, ClassNotFoundException {
		return getAllClassesInJar(Well1024a.class);
	}

	/**
	 * Obtain a list of all classes in a jar file corresponding to a referenced
	 * class.
	 * 
	 * @param classInJarFile
	 * @return list of all the commons-math3 classes in the referenced
	 *         commons-math3 jar file
	 * @throws IOException
	 *             if an IOException occurs
	 * @throws ClassNotFoundException
	 *             if a ClassNotFoundException occurs
	 */
	private static List<String> getAllClassesInJar(Class<?> classInJarFile) throws IOException, ClassNotFoundException {
		List<String> classPathNames = new ArrayList<>();
		String jarLocation = classInJarFile.getProtectionDomain().getCodeSource().getLocation().getPath();
		File f = new File(jarLocation);
		try (FileInputStream fis = new FileInputStream(f);
				JarArchiveInputStream jais = new JarArchiveInputStream(fis)) {
			while (true) {
				JarArchiveEntry jae = jais.getNextJarEntry();
				if (jae == null) {
					break;
				}
				String name = jae.getName();
				if (name.endsWith(".class")) {
					classPathNames.add(name);
				}
			}
		}

		String jarName = jarLocation.substring(jarLocation.lastIndexOf("/") + 1);
		addClassPathNamesToJarsAndClasses(jarName, classPathNames);

		return classPathNames;
	}

	/**
	 * Add jar and classes to the map of jars and their classes.
	 * 
	 * @param log4jJar
	 *            the log4j jar file
	 * @param classPathNames
	 *            the list of log4j classes
	 */
	private static void addClassPathNamesToJarsAndClasses(String jar, List<String> classPathNames) {
		for (String classPathName : classPathNames) {
			String className = classPathName.substring(0, classPathName.length() - 6);
			className = className.replace("/", ".");
			addJarAndClass(jar, className);
		}
	}

	/**
	 * Dislay all the log4j classes
	 * 
	 * @param log4jClassPathNames
	 *            the list of log4j classes
	 */
	private static void displayLog4JClassPathNames(List<String> log4jClassPathNames) {
		int numClasses = 0;
		System.out.println("\nAll log4j class files:");
		for (String classPathName : log4jClassPathNames) {
			numClasses++;
			System.out.println(" #" + numClasses + ": " + classPathName);
		}
	}

	/**
	 * Dislay all the commons-math3 classes
	 * 
	 * @param commonsMath3ClassPathNames
	 *            the list of commons-math3 classes
	 */
	private static void displayCommonsMath3ClassPathNames(List<String> commonsMath3ClassPathNames) {
		int numClasses = 0;
		System.out.println("\nAll commons-math3 class files:");
		for (String classPathName : commonsMath3ClassPathNames) {
			numClasses++;
			System.out.println(" #" + numClasses + ": " + classPathName);
		}
	}

	/**
	 * Obtain a list of all the classes that have been loaded by the
	 * classloader.
	 * 
	 * @return a list of all the classes that have been loaded by the
	 *         classloader
	 * @throws NoSuchFieldException
	 *             if NoSuchFieldException occurs
	 * @throws SecurityException
	 *             if SecurityException occurs
	 * @throws IllegalArgumentException
	 *             if IllegalArgumentException occurs
	 * @throws IllegalAccessException
	 *             if IllegalAccessException occurs
	 */
	private static List<Class<?>> getLoadedClasses()
			throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		ClassLoader cl = BuildLite.class.getClassLoader();
		Class<?> clClazz = cl.getClass();
		while (clClazz != java.lang.ClassLoader.class) {
			clClazz = clClazz.getSuperclass();
		}
		Field f = clClazz.getDeclaredField("classes");
		f.setAccessible(true);
		@SuppressWarnings("unchecked")
		Vector<Class<?>> classes = (Vector<Class<?>>) f.get(cl);
		List<Class<?>> list = new ArrayList<>(classes);

		return list;
	}

	/**
	 * Group the classes by jar file. Also, obtain the jar file sizes for
	 * comparison purposes.
	 * 
	 * @param loadedClasses
	 *            the list of loaded classes
	 */
	private static void groupLoadedClassesByJarAndClass(List<Class<?>> loadedClasses) {
		for (Class<?> clazz : loadedClasses) {
			String pathToClass = getPathToClass(clazz);
			if (pathToClass == null) {
				addJarAndClass("?", clazz.getName());
			} else if (pathToClass.endsWith(".jar")) {
				String jarName = pathToClass.substring(pathToClass.lastIndexOf("/") + 1);
				addJarAndClass(jarName, clazz.getName());

				// for comparison purposes
				if (!jarSizes.containsKey(jarName)) {
					String jarPath = pathToClass;
					File jarFile = new File(jarPath);
					long fileLength = jarFile.length();
					jarSizes.put(jarName, fileLength);
				}

			} else if (pathToClass.contains("systemml")) {
				addJarAndClass("SystemML", clazz.getName());
			} else {
				addJarAndClass("Other", clazz.getName());
			}
		}
	}

	/**
	 * Add a jar and class to the map of jars to their sets of classes.
	 * 
	 * @param jarName
	 *            the name of the jar file
	 * @param className
	 *            the name of the class
	 */
	private static void addJarAndClass(String jarName, String className) {
		if (jarsAndClasses.containsKey(jarName)) {
			SortedSet<String> classNames = jarsAndClasses.get(jarName);
			classNames.add(className);
		} else {
			SortedSet<String> classNames = new TreeSet<>();
			classNames.add(className);
			jarsAndClasses.put(jarName, classNames);
		}
	}

	/**
	 * Display the list of loaded classes.
	 * 
	 * @param loadedClasses
	 *            the list of loaded classes
	 */
	private static void displayLoadedClasses(List<Class<?>> loadedClasses) {
		int numClasses = 0;
		System.out.println("\nLoaded classes:");
		for (Class<?> clazz : loadedClasses) {
			numClasses++;
			System.out.println(" #" + numClasses + ": " + clazz + " (" + getPathToClass(clazz) + ")");
		}
	}

	/**
	 * Obtain the file system path to the location of a class.
	 * 
	 * @param clazz
	 *            the class
	 * @return the file system path to the location of a class
	 */
	private static String getPathToClass(Class<?> clazz) {
		try {
			return clazz.getProtectionDomain().getCodeSource().getLocation().getPath();
		} catch (java.lang.NullPointerException e) {
			return null;
		}
	}

	/**
	 * Display the required classes grouped by their jar files.
	 * 
	 * @throws IOException
	 *             if IOException occurs
	 */
	private static void displayJarsAndClasses() throws IOException {
		ClassLoader cl = BuildLite.class.getClassLoader();
		System.out.println("\nRequired Classes Grouped by Jar:");
		Set<String> jarNames = jarsAndClasses.keySet();
		int numClasses = 0;
		for (String jarName : jarNames) {
			SortedSet<String> classNames = jarsAndClasses.get(jarName);
			StringBuilder sb = new StringBuilder();
			int totalBytesUncompressed = 0;
			for (String className : classNames) {
				String classNamePath = className.replace(".", "/") + ".class";
				InputStream is = cl.getResourceAsStream(classNamePath);
				byte[] bytes = IOUtils.toByteArray(is);
				int numBytes = bytes.length;
				numClasses++;
				sb.append(" #" + numClasses + " " + className + " [" + NumberFormat.getInstance().format(numBytes)
						+ " bytes])\n");
				totalBytesUncompressed += numBytes;
			}
			System.out.println("Jar: " + jarName + " [" + NumberFormat.getInstance().format(totalBytesUncompressed)
					+ " bytes uncompressed]");
			System.out.println(sb.toString());
		}
	}

	/**
	 * Examine all java source files for additional classes to load. A
	 * relatively easy though not perfect way to do this is to look for imports
	 * of specified packages.
	 * 
	 * @throws IOException
	 *             if IOException occurs
	 * @throws ClassNotFoundException
	 *             if ClassNotFoundException occurs
	 */
	private static void scanJavaFilesForClassesToLoad() throws IOException, ClassNotFoundException {
		System.out.println("\nScanning java files for additional classes to load");
		int totalMatches = 0;
		SortedSet<String> uniqueMatches = new TreeSet<>();
		File base = new File(BASE_SRC_DIR);
		List<File> javaFiles = (List<File>) FileUtils.listFiles(base, new String[] { "java" }, true);
		for (File javaFile : javaFiles) {
			String content = FileUtils.readFileToString(javaFile);
			for (String additionalPackage : additionalPackages) {
				String s = "import " + additionalPackage + "(.*?);";
				Pattern p = Pattern.compile(s);
				Matcher m = p.matcher(content);
				while (m.find()) {
					totalMatches++;
					String match = m.group(1);
					String matchClass = additionalPackage + match;
					uniqueMatches.add(matchClass);
				}
			}
		}
		System.out.println("Total matches found from scan: " + totalMatches);
		int uniqueMatchesSize = uniqueMatches.size();
		System.out.println("Unique matches found from scan: " + uniqueMatchesSize);
		int numMatches = 0;
		for (String matchClass : uniqueMatches) {
			Class<?> clazz = Class.forName(matchClass);
			String pathToClass = getPathToClass(clazz);
			System.out.println(" #" + ++numMatches + ": Loaded " + clazz.getName() + " from " + pathToClass);
		}
	}

	/**
	 * Display a list of the jar dependencies twice. The first display lists the
	 * jars, which can be useful when assembling information for the LICENSE
	 * file. The second display lists the jars and their sizes, which is useful
	 * for gauging the size decrease offered by the lite jar file.
	 */
	private static void displayJarSizes() {
		System.out.println("\nIndividual Jar Dependencies (for Comparison):");
		Set<String> jarNames = jarSizes.keySet();

		for (String jarName : jarNames) {
			System.out.println(jarName);
		}
		System.out.println();

		Long totalSize = 0L;
		int count = 0;
		for (String jarName : jarNames) {
			Long jarSize = jarSizes.get(jarName);
			String jarSizeDisplay = FileUtils.byteCountToDisplaySize(jarSize);
			System.out.println(" #" + ++count + ": " + jarName + " (" + jarSizeDisplay + " ["
					+ NumberFormat.getInstance().format(jarSize) + " bytes])");
			totalSize = totalSize + jarSize;
		}
		String totalSizeDisplay = FileUtils.byteCountToDisplaySize(totalSize);
		System.out.println("Total Size of Jar Dependencies: " + totalSizeDisplay + " ["
				+ NumberFormat.getInstance().format(totalSize) + " bytes]");
	}

	/**
	 * Generate maven assembly dependency sets that can be used by the lite.xml
	 * assembly.
	 * 
	 * @throws IOException
	 *             if IOException occurs
	 */
	private static void createDependencySets() throws IOException {
		System.out.println("\nCreating maven dependency sets");

		StringBuilder sb = new StringBuilder();
		sb.append("\t<dependencySets>\n");
		Set<String> jarNames = jarsAndClasses.keySet();
		for (String jarName : jarNames) {
			String s = generateDependencySet(jarName, jarsAndClasses.get(jarName));
			sb.append(s);
		}
		sb.append(generateSystemMLDependencySet());
		sb.append("\t</dependencySets>\n");
		System.out.println(sb.toString());

		final String liteXml = "src/assembly/lite.xml";
		File f = new File(liteXml);
		if (f.exists()) {
			System.out.println("Found '" + liteXml + "', so updating dependencySets in the file.");
			String s = FileUtils.readFileToString(f);
			int start = s.indexOf("\t<dependencySets>");
			int end = s.indexOf("</dependencySets>") + "</dependencySets>".length() + 1;
			String before = s.substring(0, start);
			String after = s.substring(end);
			String newS = before + sb.toString() + after;
			FileUtils.writeStringToFile(f, newS);
		}
	}

	/**
	 * Generate a maven assembly dependency set that can be used by the lite.xml
	 * assembly. Note that additional resources can be added to the jar by the
	 * additionalJarToFileMappingsForDependencySets entries.
	 * 
	 * @param jarName
	 *            the name of the jar file
	 * @param classNames
	 *            a set of the classes in the jar file
	 * @return a string representation of the dependency set consisting of the
	 *         resources in the jar file
	 */
	private static String generateDependencySet(String jarName, SortedSet<String> classNames) {
		StringBuilder sb = new StringBuilder();
		String jarNameNoVersion = null;
		if ("SystemML".equalsIgnoreCase(jarName)) {
			jarNameNoVersion = "systemml";
			return ""; // handle in generateSystemMLDependencySet()
		} else {
			jarNameNoVersion = jarName.substring(0, jarName.lastIndexOf("-"));
		}
		sb.append("\t\t<dependencySet>\n");
		sb.append("\t\t\t<includes>\n");
		sb.append("\t\t\t\t<include>*:" + jarNameNoVersion + "</include>\n");
		sb.append("\t\t\t</includes>\n");

		sb.append("\t\t\t<unpackOptions>\n");
		sb.append("\t\t\t\t<includes>\n");

		Set<String> jarsWithAdditionalFileMappings = additionalJarToFileMappingsForDependencySets.keySet();
		if (jarsWithAdditionalFileMappings.contains(jarNameNoVersion)) {
			SortedSet<String> additionalResourceFiles = additionalJarToFileMappingsForDependencySets
					.get(jarNameNoVersion);
			for (String resourceFile : additionalResourceFiles) {
				sb.append("\t\t\t\t\t<include>" + resourceFile + "</include>\n");
			}
		}

		if (jarName.startsWith("log4j")) {
			sb.append("\t\t\t\t\t<include>**/*.class</include>\n");
		} else if (includeAllCommonsMath3 && (jarName.startsWith("commons-math3"))) {
			sb.append("\t\t\t\t\t<include>**/*.class</include>\n");
		} else {
			for (String className : classNames) {
				String classFileName = className.replace(".", "/") + ".class";
				sb.append("\t\t\t\t\t<include>" + classFileName + "</include>\n");
			}
		}
		sb.append("\t\t\t\t</includes>\n");
		sb.append("\t\t\t</unpackOptions>\n");

		sb.append("\t\t\t<scope>compile</scope>\n");
		sb.append("\t\t\t<unpack>true</unpack>\n");
		sb.append("\t\t</dependencySet>\n");
		sb.append("\n");
		return sb.toString();
	}

	private static String generateSystemMLDependencySet() {
		StringBuilder sb = new StringBuilder();
		sb.append("\t\t<dependencySet>\n");
		sb.append("\t\t\t<includes>\n");
		sb.append("\t\t\t\t<include>*:systemml*</include>\n");
		sb.append("\t\t\t</includes>\n");
		sb.append("\t\t\t<unpackOptions>\n");
		sb.append("\t\t\t\t<excludes>\n");
		sb.append("\t\t\t\t\t<exclude>META-INF/DEPENDENCIES</exclude>\n");
		sb.append("\t\t\t\t\t<exclude>META-INF/maven/**</exclude>\n");
		sb.append("\t\t\t\t\t<exclude>kernels/**</exclude>\n");
		sb.append("\t\t\t\t\t<exclude>lib/**</exclude>\n");
		sb.append("\t\t\t\t</excludes>\n");
		sb.append("\t\t\t</unpackOptions>\n");
		sb.append("\t\t\t<outputDirectory>.</outputDirectory>\n");
		sb.append("\t\t\t<scope>compile</scope>\n");
		sb.append("\t\t\t<unpack>true</unpack>\n");
		sb.append("\t\t</dependencySet>\n");
		return sb.toString();
	}
}
