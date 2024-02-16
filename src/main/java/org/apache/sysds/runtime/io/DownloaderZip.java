package org.apache.sysds.runtime.io;

import java.io.*;

import java.net.URL;
import java.util.zip.*;

public class DownloaderZip {

	public static void downloaderZip(String url, File dest, String startsWith, String endsWith) {

		try {
			ZipInputStream in = new ZipInputStream(new BufferedInputStream(new URL(url).openConnection().getInputStream()));


			ZipEntry entry;
			while((entry = in.getNextEntry()) != null) {
				StringBuilder path = new StringBuilder(dest.getPath()).append('/').append(entry.getName());
				File file = new File(path.toString());

				if(entry.isDirectory()){
					file.mkdirs();
					continue;
				}
				if(entry.getName().startsWith(startsWith) && entry.getName().endsWith(endsWith)){
					FileOutputStream out = new FileOutputStream(file);
					for (int read = in.read(); read != -1; read = in.read()){
						out.write(read);
					}
					out.close();

					// double[] z = ReaderWavFile.readMonoAudioFromWavFile(file.getPath());
					// System.out.println("hi");
				}

			}

		} catch ( IOException e){
			e.printStackTrace();
		}


	}

}
