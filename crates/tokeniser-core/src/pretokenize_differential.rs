use fancy_regex::Regex;

use crate::pretokenize::Pretokenizer;

const PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

const PIECES: &[&str] = &[
    "a", "Zed", "é", "ß", "日本", "0", "42", "٣", "²", " ", "  ", "\n", "\t", "\r\n", "\u{a0}",
    "\u{2003}", "\u{3000}", "\u{85}", "'", "'s", "'re", "'ll", "'d", "'m", "'t", "'ve", ".", ",",
    "!", "-", "_", "\u{903}", "😀", "\u{200b}", "\u{200d}", "\u{301}",
];

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn random_char(&mut self) -> char {
        loop {
            if let Some(c) = char::from_u32((self.next() % 0x11_0000) as u32) {
                return c;
            }
        }
    }
}

fn random_text(rng: &mut XorShift) -> String {
    let mut text = String::new();
    for _ in 0..rng.below(24) {
        if rng.below(4) == 0 {
            text.push(rng.random_char());
        } else {
            text.push_str(PIECES[rng.below(PIECES.len())]);
        }
    }
    text
}

fn assert_same_split(regex: &Regex, pretokenizer: &Pretokenizer, text: &str) {
    let expected: Vec<&str> = regex
        .find_iter(text)
        .map(|m| m.expect("regex match").as_str())
        .collect();
    let actual: Vec<&str> = pretokenizer.chunks(text).collect();
    assert_eq!(actual, expected, "input: {text:?}");
}

#[test]
fn matches_regex_on_random_strings() {
    let regex = Regex::new(PATTERN).unwrap();
    let pretokenizer = Pretokenizer::new();
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for _ in 0..200_000 {
        assert_same_split(&regex, &pretokenizer, &random_text(&mut rng));
    }
}

#[test]
fn matches_regex_for_every_unicode_scalar() {
    let regex = Regex::new(PATTERN).unwrap();
    let pretokenizer = Pretokenizer::new();
    for c in (0..=0x10_FFFF).filter_map(char::from_u32) {
        for text in [
            format!("a{c}b"),
            format!(" {c}"),
            format!("{c}{c} \n"),
            format!("'{c}"),
        ] {
            assert_same_split(&regex, &pretokenizer, &text);
        }
    }
}
