use std::collections::HashMap;
use std::path::Path;

use ahash::RandomState;

use crate::byte_map::byte_decoder;
use crate::error::{LoadError, UnknownToken};

pub struct Vocab {
    encoder: HashMap<Vec<u8>, u32, RandomState>,
    decoder: Vec<Vec<u8>>,
}

impl Vocab {
    pub fn from_gpt2_files(data_dir: &Path) -> Result<Self, LoadError> {
        let json = std::fs::read_to_string(data_dir.join("encoder.json"))?;
        let symbols: HashMap<String, u32> = serde_json::from_str(&json)?;
        let to_byte = byte_decoder();

        let mut encoder = HashMap::with_capacity_and_hasher(symbols.len(), RandomState::new());
        let mut decoder = vec![Vec::new(); symbols.len()];
        for (symbol, id) in symbols {
            let bytes = symbol
                .chars()
                .map(|c| {
                    to_byte
                        .get(&c)
                        .copied()
                        .ok_or(LoadError::UnknownSymbolChar(c))
                })
                .collect::<Result<Vec<u8>, _>>()?;
            *decoder
                .get_mut(id as usize)
                .ok_or(LoadError::IdOutOfRange(id))? = bytes.clone();
            encoder.insert(bytes, id);
        }
        Ok(Self { encoder, decoder })
    }

    pub fn id(&self, bytes: &[u8]) -> Option<u32> {
        self.encoder.get(bytes).copied()
    }

    pub fn bytes(&self, id: u32) -> Result<&[u8], UnknownToken> {
        self.decoder
            .get(id as usize)
            .map(Vec::as_slice)
            .ok_or(UnknownToken(id))
    }
}
