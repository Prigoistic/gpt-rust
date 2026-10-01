use crate::char_class::{CharClass, CharClasses};

const CONTRACTIONS: [&str; 7] = ["s", "t", "m", "d", "re", "ve", "ll"];

/// Splits text exactly like the GPT-2 pattern
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
pub struct Pretokenizer {
    classes: CharClasses,
}

impl Pretokenizer {
    pub fn new() -> Self {
        Self {
            classes: CharClasses::new(),
        }
    }

    pub fn chunks<'a>(&'a self, text: &'a str) -> Chunks<'a> {
        Chunks {
            pretokenizer: self,
            rest: text,
        }
    }

    fn chunk_len(&self, s: &str) -> usize {
        let mut chars = s.chars();
        let first = chars.next().expect("non-empty input");

        if first == '\'' {
            if let Some(len) = contraction_len(&s[1..]) {
                return 1 + len;
            }
        }

        let first_class = self.classes.classify(first);
        if first == ' ' {
            if let Some(next) = chars.next() {
                let class = self.classes.classify(next);
                if class != CharClass::Space {
                    return 1 + self.run_len(&s[1..], class);
                }
            }
        } else if first_class != CharClass::Space {
            return self.run_len(s, first_class);
        }
        self.whitespace_len(s)
    }

    fn run_len(&self, s: &str, class: CharClass) -> usize {
        s.char_indices()
            .find(|&(_, c)| self.classes.classify(c) != class)
            .map_or(s.len(), |(i, _)| i)
    }

    /// `\s+(?!\S)` backtracks one char when the run is followed by a non-space,
    /// leaving the last space to prefix the next chunk; a lone space stays whole.
    fn whitespace_len(&self, s: &str) -> usize {
        let mut count = 0;
        let mut last_start = 0;
        let mut end = s.len();
        for (i, c) in s.char_indices() {
            if self.classes.classify(c) != CharClass::Space {
                end = i;
                break;
            }
            last_start = i;
            count += 1;
        }
        if end == s.len() || count == 1 {
            end
        } else {
            last_start
        }
    }
}

fn contraction_len(after_quote: &str) -> Option<usize> {
    CONTRACTIONS
        .iter()
        .find(|c| after_quote.starts_with(*c))
        .map(|c| c.len())
}

pub struct Chunks<'a> {
    pretokenizer: &'a Pretokenizer,
    rest: &'a str,
}

impl<'a> Iterator for Chunks<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        let (chunk, rest) = self.rest.split_at(self.pretokenizer.chunk_len(self.rest));
        self.rest = rest;
        Some(chunk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(text: &str) -> Vec<&str> {
        let pretokenizer = Box::leak(Box::new(Pretokenizer::new()));
        pretokenizer.chunks(text).collect()
    }

    #[test]
    fn splits_words_and_punctuation() {
        assert_eq!(split("Hello, world!"), ["Hello", ",", " world", "!"]);
    }

    #[test]
    fn splits_contractions() {
        assert_eq!(split("don't"), ["don", "'t"]);
        assert_eq!(split("we'll go"), ["we", "'ll", " go"]);
    }

    #[test]
    fn attaches_one_leading_space() {
        assert_eq!(split("a  b"), ["a", " ", " b"]);
        assert_eq!(split("x 42"), ["x", " 42"]);
    }

    #[test]
    fn handles_trailing_and_mixed_whitespace() {
        assert_eq!(split("a   "), ["a", "   "]);
        assert_eq!(split("a\n\nb"), ["a", "\n", "\n", "b"]);
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(split("").is_empty());
    }
}
