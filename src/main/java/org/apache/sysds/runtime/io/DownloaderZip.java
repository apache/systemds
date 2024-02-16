package org.apache.sysds.runtime.io;

import javax.sound.sampled.*;
import java.io.*;

import java.net.URL;
import java.util.zip.*;

public class DownloaderZip {

	public static void downloaderZip(String url, File dest, String startsWith, String endsWith) {

		try {
			ZipInputStream in = new ZipInputStream(new BufferedInputStream(new URL(url).openConnection().getInputStream()));


			ZipEntry entry;
			int cnt = 0;
			while((entry = in.getNextEntry()) != null) {
				StringBuilder path = new StringBuilder(dest.getPath()).append('/').append(entry.getName());
				File file = new File(path.toString());

				if(entry.isDirectory()){
					file.mkdirs();
					continue;
				}
				if(entry.getName().startsWith(startsWith) && entry.getName().endsWith(endsWith)){

					/*
					AudioFormat format = new AudioFormat( AudioFormat.Encoding.PCM_SIGNED, 16000, 16, 1, 2, 16000, false);
					int length = (int) Math.ceil((double) entry.getExtra().length / format.getFrameSize());
					AudioInputStream audio = new AudioInputStream(new ByteArrayInputStream(entry.getExtra()), format, length);
					AudioSystem.write(audio, AudioFileFormat.Type.WAVE, file);
					*/

					FileOutputStream out = new FileOutputStream(file);
					for (int read = in.read(); read != -1; read = in.read()){
						out.write(read);
					}
					out.close();

					cnt++;
					if(cnt%50 == 0){
						System.out.println(cnt + "/8008");
					}

					// TODO: only for debugging
					if(cnt == 200){
						break;
					}
				}

			}

		} catch ( IOException e){
			e.printStackTrace();
		}


	}

}
