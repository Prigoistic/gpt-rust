mod heap;
mod scan;

use crate::vocab::Vocab;
use heap::HeapMerger;
use scan::ScanMerger;

const HEAP_THRESHOLD: usize = 128;

/// Merges one multi-token chunk into token ids, choosing the algorithm by length.
#[derive(Default)]
pub struct Merger {
    scan: ScanMerger,
    heap: HeapMerger,
}

impl Merger {
    pub fn merge(&mut self, vocab: &Vocab, chunk: &[u8], out: &mut Vec<u32>) {
        if chunk.len() < HEAP_THRESHOLD {
            self.scan.merge(vocab, chunk, out);
        } else {
            self.heap.merge(vocab, chunk, out);
        }
    }

    #[cfg(test)]
    fn merge_with(&mut self, use_heap: bool, vocab: &Vocab, chunk: &[u8], out: &mut Vec<u32>) {
        if use_heap {
            self.heap.merge(vocab, chunk, out);
        } else {
            self.scan.merge(vocab, chunk, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn vocab() -> Vocab {
        Vocab::from_gpt2_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"))
            .expect("load vocab")
    }

    fn both(vocab: &Vocab, chunk: &[u8]) -> (Vec<u32>, Vec<u32>) {
        let mut merger = Merger::default();
        let (mut scan, mut heap) = (Vec::new(), Vec::new());
        merger.merge_with(false, vocab, chunk, &mut scan);
        merger.merge_with(true, vocab, chunk, &mut heap);
        (scan, heap)
    }

    #[test]
    fn heap_matches_scan_on_repeated_patterns() {
        let vocab = vocab();
        for unit in [
            "a",
            "ab",
            "=",
            "the ",
            "ing",
            "\u{e9}",
            "supercalifragilistic",
        ] {
            for repeats in [2, 3, 7, 50, 300] {
                let chunk = unit.repeat(repeats);
                let (scan, heap) = both(&vocab, chunk.as_bytes());
                assert_eq!(heap, scan, "{unit:?} x {repeats}");
            }
        }
    }

    #[test]
    fn heap_matches_scan_on_random_bytes() {
        let vocab = vocab();
        let alphabet = b"abcde fghij.,'\n\xc3\xa9";
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..20_000 {
            let len = 2 + (next() % 200) as usize;
            let chunk: Vec<u8> = (0..len)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            let (scan, heap) = both(&vocab, &chunk);
            assert_eq!(heap, scan, "{chunk:?}");
        }
    }
}
