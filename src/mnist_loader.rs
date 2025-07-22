use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub struct MnistDataloader {
    pub training_images_filepath: String,
    pub training_labels_filepath: String,
    pub test_images_filepath: String,
    pub test_labels_filepath: String,
}

impl MnistDataloader {
    pub fn new(
        training_images_filepath: String,
        training_labels_filepath: String,
        test_images_filepath: String,
        test_labels_filepath: String,
    ) -> Self {
        Self {
            training_images_filepath,
            training_labels_filepath,
            test_images_filepath,
            test_labels_filepath,
        }
    }

    fn read_images_labels<P: AsRef<Path>>(
        &self,
        images_filepath: P,
        labels_filepath: P,
    ) -> io::Result<(Vec<Vec<u8>>, Vec<u8>)> {
        // Read labels
        let mut labels_file = BufReader::new(File::open(labels_filepath)?);
        let mut labels_header = [0u8; 8];
        labels_file.read_exact(&mut labels_header)?;
        let magic = u32::from_be_bytes([
            labels_header[0],
            labels_header[1],
            labels_header[2],
            labels_header[3],
        ]);
        let size = u32::from_be_bytes([
            labels_header[4],
            labels_header[5],
            labels_header[6],
            labels_header[7],
        ]);
        if magic != 2049 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Magic number mismatch, expected 2049, got {}", magic),
            ));
        }
        let mut labels = vec![0u8; size as usize];
        labels_file.read_exact(&mut labels)?;

        // Read images
        let mut images_file = BufReader::new(File::open(images_filepath)?);
        let mut images_header = [0u8; 16];
        images_file.read_exact(&mut images_header)?;
        let magic = u32::from_be_bytes([
            images_header[0],
            images_header[1],
            images_header[2],
            images_header[3],
        ]);
        let size = u32::from_be_bytes([
            images_header[4],
            images_header[5],
            images_header[6],
            images_header[7],
        ]);
        let rows = u32::from_be_bytes([
            images_header[8],
            images_header[9],
            images_header[10],
            images_header[11],
        ]);
        let cols = u32::from_be_bytes([
            images_header[12],
            images_header[13],
            images_header[14],
            images_header[15],
        ]);
        if magic != 2051 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Magic number mismatch, expected 2051, got {}", magic),
            ));
        }
        let mut image_data = vec![0u8; (size * rows * cols) as usize];
        images_file.read_exact(&mut image_data)?;

        let mut images = Vec::with_capacity(size as usize);
        for i in 0..size as usize {
            let start = i * (rows * cols) as usize;
            let end = start + (rows * cols) as usize;
            images.push(image_data[start..end].to_vec());
        }

        Ok((images, labels))
    }

    pub fn load_data(&self) -> io::Result<((Vec<Vec<u8>>, Vec<u8>), (Vec<Vec<u8>>, Vec<u8>))> {
        let train = self.read_images_labels(
            &self.training_images_filepath,
            &self.training_labels_filepath,
        )?;
        let test =
            self.read_images_labels(&self.test_images_filepath, &self.test_labels_filepath)?;
        Ok((train, test))
    }
}
