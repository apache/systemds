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

package org.apache.sysds.runtime.io;

import java.io.File;
import java.io.IOException;
import java.io.BufferedInputStream;
import java.io.FileOutputStream;

import java.net.URL;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class DownloaderZip {

	public static void downloaderZip(String url, File dest, String startsWith, String endsWith) {

		try {
			ZipInputStream in = new ZipInputStream(
				new BufferedInputStream(new URL(url).openConnection().getInputStream()));

			ZipEntry entry;
			int cnt = 0;
			while((entry = in.getNextEntry()) != null) {
				String path = dest.getPath() + '/' + entry.getName();
				File file = new File(path);

				if(entry.isDirectory()) {
					file.mkdirs();
					continue;
				}

				if(entry.getName().startsWith(startsWith) && entry.getName().endsWith(endsWith)) {

					/*
					 * AudioFormat format = new AudioFormat( AudioFormat.Encoding.PCM_SIGNED, 16000, 16, 1, 2, 16000,
					 * false); int length = (int) Math.ceil((double) entry.getExtra().length / format.getFrameSize());
					 * AudioInputStream audio = new AudioInputStream(new ByteArrayInputStream(entry.getExtra()), format,
					 * length); AudioSystem.write(audio, AudioFileFormat.Type.WAVE, file);
					 */

					FileOutputStream out = new FileOutputStream(file);
					for(int read = in.read(); read != -1; read = in.read()) {
						out.write(read);
					}
					out.close();

					cnt++;
					if(cnt % 50 == 0) {
						System.out.println(cnt + "/8008");
					}

					// TODO: only for debugging
					if(cnt == 200) {
						break;
					}
				}

			}

		}
		catch(IOException e) {
			e.printStackTrace();
		}

	}

}
