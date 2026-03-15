use anyhow::Result;
use memmap2::{Mmap, MmapOptions};
use std::fs::File;

pub struct Dataset {
    mmap: Mmap,
    len: usize,
    percentage: f64,
}

impl Dataset {
    pub fn new(path: &str, percentage: f64) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        #[cfg(target_os = "linux")]
        {
            let _ = mmap.advise(memmap2::Advice::Random);
            let _ = mmap.advise(memmap2::Advice::WillNeed);
        }

        let full_dataset: &[u32] = bytemuck::cast_slice(&mmap);
        let len = if percentage < 100.0 {
            ((full_dataset.len() as f64) * (percentage / 100.0)) as usize
        } else {
            full_dataset.len()
        };

        Ok(Self {
            mmap,
            len,
            percentage,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get_slice(&self) -> &[u32] {
        let full_dataset: &[u32] = bytemuck::cast_slice(&self.mmap);
        &full_dataset[..self.len]
    }

    pub fn print_stats(&self) {
        if self.percentage < 100.0 {
            println!(
                "Using {}% of the dataset! ({} tokens)",
                self.percentage, self.len
            );
        } else {
            println!("Loaded {} tokens!", self.len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_dataset_loading() -> Result<()> {
        // Create a temporary file
        let mut temp_file = NamedTempFile::new()?;

        // Write 100 mock tokens (400 bytes)
        let mock_tokens: Vec<u32> = (0..100).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&mock_tokens);
        temp_file.write_all(bytes)?;
        temp_file.flush()?;

        let file_path = temp_file.path().to_str().unwrap();

        // 1. Test 100% loading
        let dataset_100 = Dataset::new(file_path, 100.0)?;
        assert_eq!(dataset_100.len(), 100);
        assert!(!dataset_100.is_empty());
        assert_eq!(dataset_100.get_slice().len(), 100);
        assert_eq!(dataset_100.get_slice()[0], 0);
        assert_eq!(dataset_100.get_slice()[99], 99);

        // 2. Test 50% loading
        let dataset_50 = Dataset::new(file_path, 50.0)?;
        assert_eq!(dataset_50.len(), 50);
        assert_eq!(dataset_50.get_slice().len(), 50);

        // 3. Test 10% loading
        let dataset_10 = Dataset::new(file_path, 10.0)?;
        assert_eq!(dataset_10.len(), 10);
        assert_eq!(dataset_10.get_slice().len(), 10);

        Ok(())
    }
}
