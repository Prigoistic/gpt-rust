use std::collections::HashMap;
use std::ops::Range;
use std::path::Path;

use crate::bpe::Merger;
use crate::error::{LoadError, UnknownToken};
use crate::pretokenize::Pretokenizer;
use crate::vocab::Vocab;

pub struct Tokeniser {
    vocab: Vocab,
    pretokenizer: Pretokenizer,
}

impl Tokeniser {
    pub fn from_gpt2_files(data_dir: &Path) -> Result<Self, LoadError> {
        Ok(Self {
            vocab: Vocab::from_gpt2_files(data_dir)?,
            pretokenizer: Pretokenizer::new(),
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut merger = Merger::default();
        let mut cache: HashMap<&[u8], Range<usize>> = HashMap::new();
        let mut ids = Vec::with_capacity(text.len() / 4);
        for chunk in self.pretokenizer.chunks(text) {
            let bytes = chunk.as_bytes();
            if let Some(id) = self.vocab.id(bytes) {
                ids.push(id);
            } else if let Some(range) = cache.get(bytes) {
                ids.extend_from_within(range.clone());
            } else {
                let start = ids.len();
                merger.merge(&self.vocab, bytes, &mut ids);
                cache.insert(bytes, start..ids.len());
            }
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, UnknownToken> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.vocab.bytes(id)?);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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
