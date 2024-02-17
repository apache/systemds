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

import javax.sound.sampled.*;
import java.net.URL;

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
		List<int[]> waves = new ArrayList<>();
		List<String> labels = new ArrayList<>();
		loadAllData(url, waves, labels);

		// delete data
		// FileUtils.deleteDirectory(new File(sourceDirPath));

		saveToCSV("./tmp/waves", waves);
		saveToCSV("./tmp/labels", labels);
		saveToCSV("./tmp/commands", commands);

	}

	private static void loadAllData(String url, List<int[]> waves, List<String> labels) {

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

	private static byte[] getBytesZipFile(URL url) throws IOException {

		InputStream in = url.openConnection().getInputStream();
		// String zipFilePath = "./src/main/java/org/apache/sysds/runtime/matrix/data/mini_speech_commands_slimmed.zip";
		// InputStream in = new FileInputStream(zipFilePath);

		ByteArrayOutputStream out = new ByteArrayOutputStream();
		byte[] dataBuffer = new byte[1024];

		int bytesRead;
		while((bytesRead = in.read(dataBuffer, 0, 1024)) != -1) {
			out.write(dataBuffer, 0, bytesRead);
		}

		return out.toByteArray();

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

	private static void readWaveFiles(byte[] zipData, List<String> dirs, List<int[]> waves, List<String> labels)
		throws IOException {

		ZipInputStream stream = new ZipInputStream(new ByteArrayInputStream(zipData));
		ZipEntry entry;
		String dir = dirs.get(0);

		while((entry = stream.getNextEntry()) != null) {
			if(entry.getName().endsWith(".wav")) {
				if(!entry.getName().contains(dir)){
					dir = findDir(entry, dirs);
				}
				// read file
				// TODO: isn't working: we need an audioInputStream!
				AudioFormat format = new AudioFormat( AudioFormat.Encoding.PCM_SIGNED, 16000, 16, 1, 2, 16000, false);
				int length = (int) Math.ceil((double) entry.getExtra().length / format.getFrameSize());
				AudioInputStream audio = new AudioInputStream(new ByteArrayInputStream(entry.getExtra()), format, length);
				int[] data = ReaderWavFile.readMonoAudioFromWavFile(audio);
				waves.add(data);
				labels.add(dir);
			}
		}

	}

	private static String findDir(ZipEntry entry,  List<String> dirs){

		for (String dir : dirs){
			if(entry.getName().startsWith(dir)){
				return dir;
			}
		}

		return null;
	}

}
