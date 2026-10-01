use std::collections::HashMap;

pub fn byte_decoder() -> HashMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1u8..=0xAC)
        .chain(0xAEu8..=0xFF)
        .collect();
    let mut points: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();

    let mut next_point = 256;
    for b in 0..=u8::MAX {
        if !bytes.contains(&b) {
            bytes.push(b);
            points.push(next_point);
            next_point += 1;
        }
    }

    bytes
        .into_iter()
        .zip(points)
        .map(|(b, p)| (char::from_u32(p).expect("valid scalar value"), b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_every_byte_to_a_distinct_char() {
        let decoder = byte_decoder();
        assert_eq!(decoder.len(), 256);
        let mut bytes: Vec<u8> = decoder.values().copied().collect();
        bytes.sort_unstable();
        assert!(bytes.iter().copied().eq(0..=u8::MAX));
    }

    #[test]
    fn printable_ascii_maps_to_itself() {
        let decoder = byte_decoder();
        assert_eq!(decoder[&'A'], b'A');
        assert_eq!(decoder[&'~'], b'~');
    }

    #[test]
    fn space_is_shifted_into_the_private_range() {
        let decoder = byte_decoder();
        assert_eq!(decoder[&'\u{120}'], b' ');
    }
}
