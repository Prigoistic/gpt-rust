use std::cmp::Ordering;

use regex_syntax::hir::{Class, HirKind};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CharClass {
    Letter,
    Number,
    Space,
    Other,
}

struct RangeSet(Vec<(char, char)>);

impl RangeSet {
    fn from_pattern(pattern: &str) -> Self {
        match regex_syntax::parse(pattern)
            .expect("valid class")
            .into_kind()
        {
            HirKind::Class(Class::Unicode(class)) => {
                Self(class.iter().map(|r| (r.start(), r.end())).collect())
            }
            _ => unreachable!("{pattern} is a unicode class"),
        }
    }

    fn contains(&self, c: char) -> bool {
        self.0
            .binary_search_by(|&(lo, hi)| {
                if hi < c {
                    Ordering::Less
                } else if lo > c {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok()
    }
}

/// Classifies chars as `\p{L}`, `\p{N}`, `\s` or other, using the same Unicode
/// tables as the `regex` crate so results match a regex-based splitter exactly.
pub struct CharClasses {
    ascii: [CharClass; 128],
    letters: RangeSet,
    numbers: RangeSet,
    spaces: RangeSet,
}

impl CharClasses {
    pub fn new() -> Self {
        let mut classes = Self {
            ascii: [CharClass::Other; 128],
            letters: RangeSet::from_pattern(r"\p{L}"),
            numbers: RangeSet::from_pattern(r"\p{N}"),
            spaces: RangeSet::from_pattern(r"\s"),
        };
        for b in 0..128u8 {
            classes.ascii[b as usize] = classes.classify_slow(char::from(b));
        }
        classes
    }

    #[inline]
    pub fn classify(&self, c: char) -> CharClass {
        if c.is_ascii() {
            self.ascii[c as usize]
        } else {
            self.classify_slow(c)
        }
    }

    fn classify_slow(&self, c: char) -> CharClass {
        if self.letters.contains(c) {
            CharClass::Letter
        } else if self.numbers.contains(c) {
            CharClass::Number
        } else if self.spaces.contains(c) {
            CharClass::Space
        } else {
            CharClass::Other
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_ascii() {
        let classes = CharClasses::new();
        assert_eq!(classes.classify('a'), CharClass::Letter);
        assert_eq!(classes.classify('7'), CharClass::Number);
        assert_eq!(classes.classify(' '), CharClass::Space);
        assert_eq!(classes.classify('\n'), CharClass::Space);
        assert_eq!(classes.classify('\''), CharClass::Other);
    }

    #[test]
    fn classifies_unicode() {
        let classes = CharClasses::new();
        assert_eq!(classes.classify('é'), CharClass::Letter);
        assert_eq!(classes.classify('日'), CharClass::Letter);
        assert_eq!(classes.classify('\u{0663}'), CharClass::Number);
        assert_eq!(classes.classify('\u{00A0}'), CharClass::Space);
        assert_eq!(classes.classify('\u{0903}'), CharClass::Other);
        assert_eq!(classes.classify('😀'), CharClass::Other);
    }
}
