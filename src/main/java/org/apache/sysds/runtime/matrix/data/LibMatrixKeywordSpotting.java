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

package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.io.FileUtils;
import org.apache.sysds.runtime.io.ReaderWavFile;
import org.apache.sysds.runtime.io.DownloaderZip;

import java.io.File;
import java.io.IOException;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.BufferedWriter;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

public class LibMatrixKeywordSpotting {

	public static void main(String[] args) throws IOException {

		// download data
		String url = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip";
		File dest = new File("./tmp");
		String startsWith = "mini_speech_commands";
		String endsWith = ".wav";
		DownloaderZip.downloaderZip(url, dest, startsWith, endsWith);

		// zip contains command folders which contain corresponding .wav files
		// TODO: write directly into the csv
		List<int[]> waves = new ArrayList<>();
		List<Integer> labels = new ArrayList<>();
		List<String> commands = new ArrayList<>();

		String sourceDirPath = "./tmp/mini_speech_commands";
		extractData(sourceDirPath, waves, labels, commands);

		// delete data
		// FileUtils.deleteDirectory(new File(sourceDirPath));

		saveToCSV("./tmp/waves", waves);
		saveToCSV("./tmp/labels", labels);
		saveToCSV("./tmp/commands", commands);

	}

	private static void extractData(String sourceDir, List<int[]> waves, List<Integer> labels,
		List<String> commands) {

		try {

			// get directory names
			getDirectories(sourceDir, commands);

			for(String command : commands) {
				readWaveFiles(sourceDir, command, waves, labels, commands);
			}

		}
		catch(IOException e) {
			e.printStackTrace();
		}

	}

	private static void getDirectories(String sourceDir, List<String> commands) throws IOException {

		File[] subDirs = new File(sourceDir).listFiles();
		if(subDirs == null)
			return;

		for(File c : subDirs) {
			if(c.isDirectory()) {
				commands.add(c.getName());
			}
		}

	}

	private static void readWaveFiles(String sourceDir, String command, List<int[]> waves, List<Integer> labels,
		List<String> commands) {

		String path = sourceDir + '/' + command;
		File dir = new File(path);

		File[] waveFiles = dir.listFiles();
		if(waveFiles == null)
			return;

		for(File file : waveFiles) {
			waves.add(ReaderWavFile.readMonoAudioFromWavFile(file.getPath()));
			labels.add(commands.indexOf(command));
		}

	}

	private static void saveToCSV(String path, List<?> data) {

		try(PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(path)))) {

			for(Object elem : data) {
				if(elem instanceof int[]) {
					String str = Arrays.toString((int[]) elem);
					// remove brackets
					out.print(str.substring(1, str.length() - 1));
				}
				else {
					out.print(elem);
				}
				out.println();
			}

		}
		catch(IOException e) {
			e.printStackTrace();
		}

	}

}
