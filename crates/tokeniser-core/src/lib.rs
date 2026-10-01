use std::collections::HashMap;
use std::path::Path;

use dashmap::DashMap;
use fancy_regex::Regex;
use rayon::prelude::*;

/// GPT-2's pre-tokenization pattern. Merges never cross a match boundary, so this
/// must be identical to the original for token ids to line up with the pretrained
/// vocab. `fancy-regex` is required because of the `(?!\S)` lookahead, which the
/// linear-time `regex` crate does not support.
const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// GPT-2's byte -> printable-char table. `encoder.json` stores each token as a string
/// in this alphabet, so it is only needed at load time to turn those strings back
/// into raw bytes. All runtime work stays on bytes.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    // Bytes that are already printable map to themselves.
    let mut byte_values: Vec<u32> = Vec::new();
    for &(lo, hi) in &[(b'!', b'~'), (0xA1, 0xAC), (0xAE, 0xFF)] {
        byte_values.extend((lo as u32)..=(hi as u32));
    }

    // The remaining control/whitespace bytes are shifted to unused code points
    // starting at U+0100, so every byte gets a distinct printable stand-in.
    let mut unicode_points = byte_values.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !byte_values.contains(&b) {
            byte_values.push(b);
            unicode_points.push(256 + n);
            n += 1;
        }
    }

    byte_values
        .into_iter()
        .map(|b| b as u8)
        .zip(
            unicode_points
                .into_iter()
                .map(|c| char::from_u32(c).unwrap()),
        )
        .collect()
}

fn build_byte_decoder() -> HashMap<char, u8> {
    bytes_to_unicode()
        .into_iter()
        .map(|(b, c)| (c, b))
        .collect()
}

/// GPT-2 byte-level BPE tokeniser operating directly on raw bytes.
pub struct Tokeniser {
    /// Token bytes -> id. A GPT-2 id doubles as the merge rank, because ids were
    /// assigned in merge order. Every merge result is itself a vocab entry, so no
    /// separate (pair -> rank) table is needed.
    encoder: HashMap<Vec<u8>, u32>,
    /// Id -> token bytes, indexed by id.
    decoder: Vec<Vec<u8>>,
    pattern: Regex,
    /// Memoizes the final ids per pre-tokenized chunk. Real text repeats words
    /// constantly, and `DashMap` shards its locks so parallel chunk workers do not
    /// all contend on a single lock.
    cache: DashMap<Vec<u8>, Vec<u32>>,
}

impl Tokeniser {
    /// Loads `encoder.json` from `data_dir`. Panics if the file is missing or malformed.
    pub fn from_gpt2_files(data_dir: &Path) -> Self {
        let raw_json = std::fs::read_to_string(data_dir.join("encoder.json"))
            .expect("failed to read encoder.json");
        let raw: HashMap<String, u32> =
            serde_json::from_str(&raw_json).expect("invalid encoder.json");

        let byte_decoder = build_byte_decoder();

        let mut encoder: HashMap<Vec<u8>, u32> = HashMap::with_capacity(raw.len());
        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); raw.len()];

        for (symbol, id) in raw {
            let bytes: Vec<u8> = symbol
                .chars()
                .map(|c| {
                    *byte_decoder
                        .get(&c)
                        .expect("symbol char not in byte decoder")
                })
                .collect();
            decoder[id as usize] = bytes.clone();
            encoder.insert(bytes, id);
        }

        let pattern = Regex::new(PATTERN).expect("pattern must compile");

        Tokeniser {
            encoder,
            decoder,
            pattern,
            cache: DashMap::new(),
        }
    }

    /// Splits `text` with the GPT-2 pattern, then BPE-encodes the chunks in parallel.
    /// `par_iter` over an indexed collection keeps the output in input order.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let chunks: Vec<&str> = self
            .pattern
            .find_iter(text)
            .map(|m| m.expect("regex match failed").as_str())
            .collect();

        chunks
            .par_iter()
            .map(|chunk| self.encode_chunk(chunk.as_bytes()))
            .flatten()
            .collect()
    }

    /// Greedy BPE over one pre-tokenized chunk: repeatedly merge the adjacent pair
    /// whose combined bytes have the lowest rank until no merge is possible.
    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        if let Some(cached) = self.cache.get(chunk) {
            return cached.clone();
        }

        // Symbols are tracked as boundary indices into `chunk` instead of owned
        // strings: the i-th symbol is chunk[boundaries[i]..boundaries[i+1]]. Merging
        // two neighbours is then just removing the boundary between them, and every
        // candidate lookup below is a borrowed slice, so the loop does not allocate.
        // Starts with one symbol per byte; all 256 single bytes are in the vocab.
        let mut boundaries: Vec<usize> = (0..=chunk.len()).collect();

        loop {
            if boundaries.len() <= 2 {
                break;
            }

            // Scanning every pair each round is O(n^2) per chunk, but chunks are
            // short words, where a linear scan is faster than a heap in practice.
            let mut best_rank = u32::MAX;
            let mut best_i = None;
            for i in 0..boundaries.len() - 2 {
                let candidate = &chunk[boundaries[i]..boundaries[i + 2]];
                if let Some(&rank) = self.encoder.get(candidate) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }

            match best_i {
                Some(i) => {
                    boundaries.remove(i + 1);
                }
                None => break,
            }
        }

        // Safe to unwrap: single bytes are always in the vocab, and every merged
        // span was only formed after a successful vocab lookup above.
        let ids: Vec<u32> = (0..boundaries.len() - 1)
            .map(|i| {
                let symbol = &chunk[boundaries[i]..boundaries[i + 1]];
                *self
                    .encoder
                    .get(symbol)
                    .expect("every final symbol must be a valid vocab entry")
            })
            .collect();

        self.cache.insert(chunk.to_vec(), ids.clone());
        ids
    }

    /// Concatenates token bytes and decodes as UTF-8. Invalid sequences become U+FFFD,
    /// since a token boundary can split a multi-byte character (e.g. in streamed output).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.decoder[id as usize]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokeniser() -> Tokeniser {
        let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        Tokeniser::from_gpt2_files(&data_dir)
    }

    #[test]
    fn matches_python_reference_ascii() {
        let tok = tokeniser();
        assert_eq!(tok.encode("Hello, world!"), vec![15496, 11, 995, 0]);
        assert_eq!(
            tok.encode("don't stop believing"),
            vec![9099, 470, 2245, 14773]
        );
    }

    #[test]
    fn matches_python_reference_unicode() {
        let tok = tokeniser();
        let expected: Vec<u32> = vec![
            46903, 1098, 25, 40304, 11, 41492, 11, 10545, 245, 98, 17312, 105, 45739, 252, 11,
            30325, 222, 8582, 248, 222,
        ];
        assert_eq!(tok.encode("unicode: café, naïve, 日本語, 😀🚀"), expected);
    }

    #[test]
    fn roundtrip_decode() {
        let tok = tokeniser();
        let text = "Hello, world! don't stop believing";
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn cache_hits_on_repeated_chunk() {
        let tok = tokeniser();
        let ids1 = tok.encode("the the the");
        let ids2 = tok.encode("the the the");
        assert_eq!(ids1, ids2);
    }
}
