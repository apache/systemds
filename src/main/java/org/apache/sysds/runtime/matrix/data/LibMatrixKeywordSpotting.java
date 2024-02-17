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

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;

import java.io.IOException;
import java.io.FileInputStream;
import java.io.ByteArrayInputStream;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.BufferedWriter;

import java.util.List;
import java.util.Arrays;
import java.util.Arrays;
import java.util.ArrayList;

public class LibMatrixKeywordSpotting {

	/**
	 * Please download the
	 * <a href="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip">zip file</a> before
	 * running. Save it in "./tmp".
	 */
	public static void main(String[] args) {

		String basePath = "./tmp/";
		String zipPath = basePath + "mini_speech_commands.zip";

		try {
			// get zip data
			ZipInputStream zipStream = new ZipInputStream(new FileInputStream(zipPath));
			saveDataToCSV(basePath, zipStream);
		}
		catch(IOException e) {
			e.printStackTrace();
		}

	}

	private static void saveDataToCSV(String basePath, ZipInputStream zipStream) throws IOException {

		PrintWriter commandsCSV = new PrintWriter(new BufferedWriter(new FileWriter(basePath + "commands")));
		PrintWriter wavesCSV = new PrintWriter(new BufferedWriter(new FileWriter(basePath + "waves")));
		PrintWriter labelsCSV = new PrintWriter(new BufferedWriter(new FileWriter(basePath + "labels")));

		List<String> commands = new ArrayList<>();

		// exclude main directory
		ZipEntry entry = zipStream.getNextEntry();

		if(entry == null)
			return;
		String mainDir = entry.getName();

		while((entry = zipStream.getNextEntry()) != null) {

			if(entry.isDirectory()) {

				String dir = entry.getName();
				// remove "/" at the end
				String name = dir.substring(mainDir.length(), dir.length() - 1);

				commands.add(name);
				// save to csv
				commandsCSV.print(name);
				commandsCSV.println();

			}
			else if(isWavFileToProcess(entry)) {

				// read file
				AudioFormat format = new AudioFormat(AudioFormat.Encoding.PCM_SIGNED, 16000, 16, 1, 2, 16000, false);
				int length = (int) Math.ceil((double) entry.getExtra().length / format.getFrameSize());
				AudioInputStream audio = new AudioInputStream(new ByteArrayInputStream(entry.getExtra()), format,
					length);
				int[] data = ReaderWavFile.readMonoAudioFromWavFile(audio);

				// save to csv
				String str = Arrays.toString(data);
				wavesCSV.print(str.substring(1, str.length() - 1));
				wavesCSV.println();

				labelsCSV.print(commands.indexOf(getCommand(entry)));
				labelsCSV.println();
			}
		}

		commandsCSV.close();
		labelsCSV.close();
		wavesCSV.close();

	}

	private static boolean isWavFileToProcess(ZipEntry entry) {

		String path = entry.getName();

		if(!path.endsWith(".wav"))
			return false;

		int end = path.lastIndexOf('/');
		String file = path.substring(end + 1);

		return !file.startsWith(".");
	}

	private static String getCommand(ZipEntry entry) {

		String path = entry.getName();

		int end = path.lastIndexOf('/');
		int start = path.substring(0, end).indexOf('/');

		return path.substring(start + 1, end);
	}

}
