package com.ibm.bi.dml.test.utils;

import java.io.File;
import java.io.FilenameFilter;
import java.util.Collection;
import java.util.Vector;

/**
 * Uploads all files from subfolder /test/scripts/ except those ending with .svn
 * to HDFS under /user/hadoop/...<br/>
 * Currently used for ant auto testing.
 * 
 */
public class HDFSFileUploader {
	private static FilenameFilter filter = new FilenameFilter() {
		@Override
		public boolean accept(File dir, String name) {
			return (dir.toString().startsWith("./test/scripts/") && !dir.toString().contains(".svn") && !name
					.contains(".svn"));
		}
	};

	public static void main(String[] args) {
		Collection<File> files = listFiles(new File("."), filter, true);

		for (File file : files) {
			String cmd = "";
			try {
				cmd = "hadoop fs -put " + file.toString() + " " + "/user/hadoop" + file.toString().substring(1);
				Runtime.getRuntime().exec(cmd);
				System.out.println(cmd + " DONE");
			} catch (Exception e) {
				System.out.println(cmd + " FAILED");
			}
		}
	}

	private static Collection<File> listFiles(File directory, FilenameFilter filter, boolean recurse) {
		// List of files / directories
		Vector<File> files = new Vector<File>();

		// Get files / directories in the directory
		File[] entries = directory.listFiles();

		// Go over entries
		for (File entry : entries) {
			// If there is no filter or the filter accepts the
			// file / directory, add it to the list
			if (filter == null || filter.accept(directory, entry.getName())) {
				files.add(entry);
			}

			// If the file is a directory and the recurse flag
			// is set, recurse into the directory
			if (recurse && entry.isDirectory()) {
				files.addAll(listFiles(entry, filter, recurse));
			}
		}

		// Return collection of files
		return files;
	}
}
