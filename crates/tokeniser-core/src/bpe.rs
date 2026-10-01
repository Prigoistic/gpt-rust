use crate::vocab::Vocab;

const NO_MERGE: u32 = u32::MAX;

/// Reusable scratch space for merging one multi-token chunk into token ids.
///
/// Symbols are byte ranges of the chunk, stored as boundary indices. A GPT-2 id
/// doubles as the merge rank, so the rank of merging two neighbours is the id of
/// their concatenated bytes (if that is a vocab entry).
#[derive(Default)]
pub struct Merger {
    boundaries: Vec<usize>,
    ranks: Vec<u32>,
}

impl Merger {
    pub fn merge(&mut self, vocab: &Vocab, chunk: &[u8], out: &mut Vec<u32>) {
        self.boundaries.clear();
        self.boundaries.extend(0..=chunk.len());
        self.ranks.clear();
        self.ranks
            .extend((0..chunk.len() - 1).map(|i| pair_rank(vocab, chunk, &self.boundaries, i)));

        while let Some((i, _)) = self.best_merge() {
            self.boundaries.remove(i + 1);
            self.ranks.remove(i);
            if i < self.ranks.len() {
                self.ranks[i] = pair_rank(vocab, chunk, &self.boundaries, i);
            }
            if i > 0 {
                self.ranks[i - 1] = pair_rank(vocab, chunk, &self.boundaries, i - 1);
            }
        }

        out.extend(
            self.boundaries
                .windows(2)
                .map(|w| vocab.id(&chunk[w[0]..w[1]]).expect("symbol in vocab")),
        );
    }

    fn best_merge(&self) -> Option<(usize, u32)> {
        self.ranks
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, rank)| rank != NO_MERGE)
            .min_by_key(|&(_, rank)| rank)
    }
}

fn pair_rank(vocab: &Vocab, chunk: &[u8], boundaries: &[usize], i: usize) -> u32 {
    vocab
        .id(&chunk[boundaries[i]..boundaries[i + 2]])
        .unwrap_or(NO_MERGE)
}
