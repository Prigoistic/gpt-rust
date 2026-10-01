use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::vocab::Vocab;

const NONE: u32 = u32::MAX;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    rank: u32,
    left: u32,
    left_version: u32,
    right: u32,
    right_version: u32,
}

/// Merges long chunks in O(n log n): symbols form a linked list, candidate merges
/// live in a min-heap, and stale heap entries are discarded lazily on pop.
#[derive(Default)]
pub struct HeapMerger {
    start: Vec<usize>,
    prev: Vec<u32>,
    next: Vec<u32>,
    version: Vec<u32>,
    alive: Vec<bool>,
    heap: BinaryHeap<Reverse<Candidate>>,
}

impl HeapMerger {
    pub fn merge(&mut self, vocab: &Vocab, chunk: &[u8], out: &mut Vec<u32>) {
        self.reset(chunk.len());
        for left in 0..chunk.len() as u32 - 1 {
            self.push_pair(vocab, chunk, left, left + 1);
        }

        while let Some(Reverse(c)) = self.heap.pop() {
            if !self.is_current(&c) {
                continue;
            }
            let (left, right) = (c.left, c.right);
            let after = self.next[right as usize];
            self.next[left as usize] = after;
            if after != NONE {
                self.prev[after as usize] = left;
            }
            self.alive[right as usize] = false;
            self.version[left as usize] += 1;

            let before = self.prev[left as usize];
            if before != NONE {
                self.push_pair(vocab, chunk, before, left);
            }
            if after != NONE {
                self.push_pair(vocab, chunk, left, after);
            }
        }

        let mut symbol = 0;
        while symbol != NONE {
            let span = self.start[symbol as usize]..self.end(chunk.len(), symbol);
            out.push(vocab.id(&chunk[span]).expect("symbol in vocab"));
            symbol = self.next[symbol as usize];
        }
    }

    fn reset(&mut self, len: usize) {
        self.start.clear();
        self.start.extend(0..len);
        self.prev.clear();
        self.prev.extend((0..len as u32).map(|i| i.wrapping_sub(1)));
        self.next.clear();
        self.next
            .extend((0..len as u32).map(|i| if i as usize + 1 == len { NONE } else { i + 1 }));
        self.version.clear();
        self.version.resize(len, 0);
        self.alive.clear();
        self.alive.resize(len, true);
        self.heap.clear();
    }

    fn end(&self, len: usize, symbol: u32) -> usize {
        match self.next[symbol as usize] {
            NONE => len,
            next => self.start[next as usize],
        }
    }

    fn push_pair(&mut self, vocab: &Vocab, chunk: &[u8], left: u32, right: u32) {
        let span = self.start[left as usize]..self.end(chunk.len(), right);
        if let Some(rank) = vocab.id(&chunk[span]) {
            self.heap.push(Reverse(Candidate {
                rank,
                left,
                left_version: self.version[left as usize],
                right,
                right_version: self.version[right as usize],
            }));
        }
    }

    fn is_current(&self, c: &Candidate) -> bool {
        self.alive[c.left as usize]
            && self.alive[c.right as usize]
            && self.next[c.left as usize] == c.right
            && self.version[c.left as usize] == c.left_version
            && self.version[c.right as usize] == c.right_version
    }
}
