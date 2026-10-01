use std::path::Path;

use dashmap::DashMap;
use rayon::prelude::*;

use crate::error::{LoadError, UnknownToken};
use crate::pretokenize::Pretokenizer;
use crate::vocab::Vocab;

pub struct Tokeniser {
    vocab: Vocab,
    pretokenizer: Pretokenizer,
    cache: DashMap<Vec<u8>, Vec<u32>>,
}

impl Tokeniser {
    pub fn from_gpt2_files(data_dir: &Path) -> Result<Self, LoadError> {
        Ok(Self {
            vocab: Vocab::from_gpt2_files(data_dir)?,
            pretokenizer: Pretokenizer::new(),
            cache: DashMap::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chunks: Vec<&str> = self.pretokenizer.chunks(text).collect();
        chunks
            .par_iter()
            .flat_map_iter(|chunk| self.encode_chunk(chunk.as_bytes()))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, UnknownToken> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.vocab.bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        if let Some(cached) = self.cache.get(chunk) {
            return cached.clone();
        }

        let mut boundaries: Vec<usize> = (0..=chunk.len()).collect();
        while boundaries.len() > 2 {
            let best = (0..boundaries.len() - 2)
                .filter_map(|i| {
                    let pair = &chunk[boundaries[i]..boundaries[i + 2]];
                    self.vocab.id(pair).map(|rank| (rank, i))
                })
                .min();
            match best {
                Some((_, i)) => {
                    boundaries.remove(i + 1);
                }
                None => break,
            }
        }

        let ids: Vec<u32> = boundaries
            .windows(2)
            .map(|w| self.vocab.id(&chunk[w[0]..w[1]]).expect("symbol in vocab"))
            .collect();
        self.cache.insert(chunk.to_vec(), ids.clone());
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokeniser() -> Tokeniser {
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        Tokeniser::from_gpt2_files(&data_dir).expect("load vocab")
    }

    #[test]
    fn matches_python_reference_ascii() {
        let tok = tokeniser();
        assert_eq!(tok.encode("Hello, world!"), [15496, 11, 995, 0]);
        assert_eq!(tok.encode("don't stop believing"), [9099, 470, 2245, 14773]);
    }

    #[test]
    fn matches_python_reference_unicode() {
        let expected = [
            46903, 1098, 25, 40304, 11, 41492, 11, 10545, 245, 98, 17312, 105, 45739, 252, 11,
            30325, 222, 8582, 248, 222,
        ];
        assert_eq!(
            tokeniser().encode("unicode: café, naïve, 日本語, 😀🚀"),
            expected
        );
    }

    #[test]
    fn roundtrips() {
        let tok = tokeniser();
        let text = "Hello, world! don't stop believing";
        assert_eq!(tok.decode(&tok.encode(text)).unwrap(), text);
    }

    #[test]
    fn empty_input_encodes_to_nothing() {
        assert!(tokeniser().encode("").is_empty());
    }

    #[test]
    fn decode_rejects_unknown_ids() {
        assert!(tokeniser().decode(&[u32::MAX]).is_err());
    }
}
