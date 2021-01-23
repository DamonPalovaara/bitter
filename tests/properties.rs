use bitter::{BigEndianReader, BitReader, LittleEndianReader};
use quickcheck_macros::quickcheck;

#[quickcheck]
fn read_bytes_eq(k1: u8, data: Vec<u8>) -> bool {
    let mut bits = LittleEndianReader::new(data.as_slice());
    let mut buf = vec![0u8; usize::from(k1)];
    if !bits.read_bytes(&mut buf) {
        k1 > data.len() as u8
    } else {
        buf.as_slice() == &data[..usize::from(k1)]
    }
}

#[quickcheck]
fn has_bits_remaining_bit_reads(data: Vec<u8>) -> bool {
    let mut bits = LittleEndianReader::new(data.as_slice());
    (1..128).all(|_x| bits.has_bits_remaining(1) == bits.read_bit().is_some())
}

#[quickcheck]
fn has_bits_remaining_be_bit_reads(data: Vec<u8>) -> bool {
    let mut bits = BigEndianReader::new(data.as_slice());
    (1..128).all(|_x| bits.has_bits_remaining(1) == bits.read_bit().is_some())
}

#[quickcheck]
fn has_bits_remaining(data: Vec<u8>) -> bool {
    let mut bits = LittleEndianReader::new(data.as_slice());
    (1..32).all(|x| bits.has_bits_remaining(x) == bits.read_bits(x as i32).is_some())
        && bits.has_bits_remaining(8) == bits.read_u8().is_some()
        && bits.has_bits_remaining(16) == bits.read_u16().is_some()
        && bits.has_bits_remaining(32) == bits.read_u32().is_some()
        && bits.has_bits_remaining(64) == bits.read_u64().is_some()
}

#[quickcheck]
fn read_bytes_bits(data: Vec<u8>) -> bool {
    if data.len() <= 1 {
        return true;
    }

    let mut bits = LittleEndianReader::new(data.as_slice());
    bits.read_bits_unchecked(3);

    let mut buf = vec![0u8; data.len() - 1];
    bits.read_bytes(&mut buf)
}

#[quickcheck]
fn read_byte_eq(data: Vec<u8>) -> bool {
    let mut lebits = LittleEndianReader::new(data.as_slice());
    let mut bebits = BigEndianReader::new(data.as_slice());

    for _ in &data {
        if lebits.read_u8() != bebits.read_u8() {
            return false;
        }
    }

    lebits.is_empty() && bebits.is_empty()
}

#[quickcheck]
fn read_byte_unchecked_eq(data: Vec<u8>) -> bool {
    let mut lebits = LittleEndianReader::new(data.as_slice());
    let mut bebits = BigEndianReader::new(data.as_slice());

    for _ in &data {
        if lebits.read_u8_unchecked() != bebits.read_u8_unchecked() {
            return false;
        }
    }

    lebits.is_empty() && bebits.is_empty()
}

#[quickcheck]
fn read_byte_alternate(data: Vec<u8>) -> bool {
    let mut lebits = LittleEndianReader::new(data.as_slice());
    let mut bebits = BigEndianReader::new(data.as_slice());

    for (i, x) in data.iter().enumerate() {
        if i % 3 == 0 {
            let val = lebits.read_u8().unwrap();
            if val != *x || val != bebits.read_u8().unwrap() {
                return false;
            }
        } else {
            let val = lebits.read_u8_unchecked();
            if val != *x || val != bebits.read_u8_unchecked() {
                return false;
            }
        }
    }

    lebits.is_empty() && bebits.is_empty()
}

#[quickcheck]
fn read_u64_le_alternate(data: Vec<u64>) -> bool {
    let mut ledata: Vec<u8> = Vec::new();
    for x in &data {
        ledata.extend_from_slice(&x.to_le_bytes());
    }
    let mut lebits = LittleEndianReader::new(ledata.as_slice());
    for _ in 0..64 {
        lebits.read_bit();
    }

    for (i, x) in data.iter().skip(1).enumerate() {
        if i % 3 == 0 {
            if lebits.read_u64().unwrap() != *x {
                return false;
            }
        } else {
            if lebits.read_u64_unchecked() != *x {
                return false;
            }
        }
    }

    lebits.is_empty()
}

#[quickcheck]
fn read_u64_be_alternate(data: Vec<u64>) -> bool {
    let mut bedata: Vec<u8> = Vec::new();
    for x in &data {
        bedata.extend_from_slice(&x.to_be_bytes());
    }
    let mut bebits = BigEndianReader::new(bedata.as_slice());
    for _ in 0..64 {
        bebits.read_bit();
    }

    for (i, x) in data.iter().skip(1).enumerate() {
        if i % 3 == 0 {
            if bebits.read_u64().unwrap() != *x {
                return false;
            }
        } else {
            if bebits.read_u64_unchecked() != *x {
                return false;
            }
        }
    }

    bebits.is_empty()
}

#[quickcheck]
fn read_u16_eq(x: u16) -> bool {
    let le_data = x.to_le_bytes();
    let be_data = x.to_be_bytes();
    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u16() == bebits.read_u16()
}

#[quickcheck]
fn read_u32_eq(x: u32) -> bool {
    let le_data = x.to_le_bytes();
    let be_data = x.to_be_bytes();
    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u32() == bebits.read_u32()
}

#[quickcheck]
fn read_u64_eq(x: u64) -> bool {
    let le_data = x.to_le_bytes();
    let be_data = x.to_be_bytes();
    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u64() == bebits.read_u64()
}

#[quickcheck]
fn read_u16_unchecked_eq(x: u16) -> bool {
    let le_data = x.to_le_bytes();
    let be_data = x.to_be_bytes();
    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u16_unchecked() == bebits.read_u16_unchecked()
}

#[quickcheck]
fn read_u32_unchecked_eq(x: u32) -> bool {
    let le_data = x.to_le_bytes();
    let be_data = x.to_be_bytes();
    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u32_unchecked() == bebits.read_u32_unchecked()
}

#[quickcheck]
fn read_u64_unchecked_eq(x: u64) -> bool {
    let le_data = x.to_le_bytes();
    let be_data = x.to_be_bytes();
    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u64_unchecked() == bebits.read_u64_unchecked()
}

#[quickcheck]
fn has_bits_remaining_bit_reads_ends(reads: u8, data: Vec<u8>) -> bool {
    fn test_fn<B: BitReader>(bits: &mut B, reads: u8) -> bool {
        let mut result = true;
        if bits.has_bits_remaining(usize::from(reads)) {
            for _ in 0..reads {
                result &= bits.read_bit().is_some();
            }
        }

        result
    }

    let mut lebits = LittleEndianReader::new(data.as_slice());
    let mut bebits = LittleEndianReader::new(data.as_slice());

    test_fn(&mut lebits, reads) && test_fn(&mut bebits, reads)
}

#[quickcheck]
fn read_u32_val_eq(bits: u32) -> bool {
    let le_data = bits.to_le_bytes();
    let be_data = bits.to_be_bytes();

    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_u32() == bebits.read_u32()
}

#[quickcheck]
fn read_u32_unchecked_val_eq(bits: u32) -> bool {
    let le_data = bits.to_le_bytes();
    let be_data = bits.to_be_bytes();

    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    let le = lebits.read_u32_unchecked();
    le == bebits.read_u32_unchecked() && le == bits
}

#[quickcheck]
fn read_f32_eq(bits: u32) -> bool {
    let le_data = bits.to_le_bytes();
    let be_data = bits.to_be_bytes();

    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_f32() == bebits.read_f32()
}

#[quickcheck]
fn read_f32_unchecked_eq(bits: u32) -> bool {
    let le_data = bits.to_le_bytes();
    let be_data = bits.to_be_bytes();

    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_f32_unchecked() == bebits.read_f32_unchecked()
}

#[quickcheck]
fn read_f64_eq(bits: u64) -> bool {
    let le_data = bits.to_le_bytes();
    let be_data = bits.to_be_bytes();

    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_f64() == bebits.read_f64()
}

#[quickcheck]
fn read_f64_unchecked_eq(bits: u64) -> bool {
    let le_data = bits.to_le_bytes();
    let be_data = bits.to_be_bytes();

    let mut lebits = LittleEndianReader::new(&le_data);
    let mut bebits = BigEndianReader::new(&be_data);

    lebits.read_f64_unchecked() == bebits.read_f64_unchecked()
}

#[quickcheck]
fn back_to_back_le_u64(x: u64, y: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_le_bytes()));
    data.extend_from_slice(&(y.to_le_bytes()));
    let mut bits = LittleEndianReader::new(data.as_slice());
    bits.read_u64() == Some(x) && bits.read_u64() == Some(y)
}

#[quickcheck]
fn back_to_back_le_unchecked_u64(x: u64, y: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_le_bytes()));
    data.extend_from_slice(&(y.to_le_bytes()));
    let mut bits = LittleEndianReader::new(data.as_slice());
    bits.read_u64_unchecked() == x && bits.read_u64_unchecked() == y
}

#[quickcheck]
fn back_to_back_be_u64(x: u64, y: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_be_bytes()));
    data.extend_from_slice(&(y.to_be_bytes()));
    let mut bits = BigEndianReader::new(data.as_slice());
    bits.read_u64() == Some(x) && bits.read_u64() == Some(y)
}

#[quickcheck]
fn back_to_back_be_unchecked_u64(x: u64, y: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_be_bytes()));
    data.extend_from_slice(&(y.to_be_bytes()));
    let mut bits = BigEndianReader::new(data.as_slice());
    bits.read_u64_unchecked() == x && bits.read_u64_unchecked() == y
}

#[quickcheck]
fn read_le_bits_64(x: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_le_bytes()));
    let mut lebits = LittleEndianReader::new(data.as_slice());
    lebits.read_bits(64) == Some(x)
}

#[quickcheck]
fn read_be_bits_64(x: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_be_bytes()));
    let mut bebits = BigEndianReader::new(data.as_slice());
    bebits.read_bits(64) == Some(x)
}

#[quickcheck]
fn read_le_bits_64_unchecked(x: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_le_bytes()));
    let mut lebits = LittleEndianReader::new(data.as_slice());
    lebits.read_bits_unchecked(64) == x
}

#[quickcheck]
fn read_be_bits_64_unchecked(x: u64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(x.to_be_bytes()));
    let mut bebits = BigEndianReader::new(data.as_slice());
    bebits.read_bits_unchecked(64) == x
}

#[quickcheck]
fn read_le_signed_bits(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_le_bytes()));
    data.extend_from_slice(&(b.to_le_bytes()));
    data.extend_from_slice(&(c.to_le_bytes()));
    data.extend_from_slice(&(d.to_le_bytes()));
    let mut lebits = LittleEndianReader::new(data.as_slice());

    lebits.read_signed_bits(8).map(|x| x as i8) == Some(a)
        && lebits.read_signed_bits(16).map(|x| x as i16) == Some(b)
        && lebits.read_signed_bits(32).map(|x| x as i32) == Some(c)
        && lebits.read_signed_bits(64).map(|x| x as i64) == Some(d)
}

#[quickcheck]
fn read_le_signed_unchecked_bits(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_le_bytes()));
    data.extend_from_slice(&(b.to_le_bytes()));
    data.extend_from_slice(&(c.to_le_bytes()));
    data.extend_from_slice(&(d.to_le_bytes()));
    let mut lebits = LittleEndianReader::new(data.as_slice());

    lebits.read_signed_bits_unchecked(8) as i8 == a
        && lebits.read_signed_bits_unchecked(16) as i16 == b
        && lebits.read_signed_bits_unchecked(32) as i32 == c
        && lebits.read_signed_bits_unchecked(64) as i64 == d
}

#[quickcheck]
fn read_le_signed_bits2(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_le_bytes()));
    data.extend_from_slice(&(b.to_le_bytes()));
    data.extend_from_slice(&(c.to_le_bytes()));
    data.extend_from_slice(&(d.to_le_bytes()));
    let mut lebits = LittleEndianReader::new(data.as_slice());
    let mut lebits2 = LittleEndianReader::new(data.as_slice());

    lebits.read_signed_bits(8).map(|x| x as i8) == lebits2.read_i8()
        && lebits.read_signed_bits(16).map(|x| x as i16) == lebits2.read_i16()
        && lebits.read_signed_bits(32).map(|x| x as i32) == lebits2.read_i32()
        && lebits.read_signed_bits(64).map(|x| x as i64) == lebits2.read_i64()
}

#[quickcheck]
fn read_le_signed_unchecked_bits2(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_le_bytes()));
    data.extend_from_slice(&(b.to_le_bytes()));
    data.extend_from_slice(&(c.to_le_bytes()));
    data.extend_from_slice(&(d.to_le_bytes()));
    let mut lebits = LittleEndianReader::new(data.as_slice());
    let mut lebits2 = LittleEndianReader::new(data.as_slice());

    lebits.read_signed_bits_unchecked(8) as i8 == lebits2.read_i8_unchecked()
        && lebits.read_signed_bits_unchecked(16) as i16 == lebits2.read_i16_unchecked()
        && lebits.read_signed_bits_unchecked(32) as i32 == lebits2.read_i32_unchecked()
        && lebits.read_signed_bits_unchecked(64) as i64 == lebits2.read_i64_unchecked()
}

#[quickcheck]
fn read_be_signed_bits(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_be_bytes()));
    data.extend_from_slice(&(b.to_be_bytes()));
    data.extend_from_slice(&(c.to_be_bytes()));
    data.extend_from_slice(&(d.to_be_bytes()));
    let mut bebits = BigEndianReader::new(data.as_slice());

    bebits.read_signed_bits(8).map(|x| x as i8) == Some(a)
        && bebits.read_signed_bits(16).map(|x| x as i16) == Some(b)
        && bebits.read_signed_bits(32).map(|x| x as i32) == Some(c)
        && bebits.read_signed_bits(64).map(|x| x as i64) == Some(d)
}

#[quickcheck]
fn read_be_signed_unchecked_bits(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_be_bytes()));
    data.extend_from_slice(&(b.to_be_bytes()));
    data.extend_from_slice(&(c.to_be_bytes()));
    data.extend_from_slice(&(d.to_be_bytes()));
    let mut bebits = BigEndianReader::new(data.as_slice());

    bebits.read_signed_bits_unchecked(8) as i8 == a
        && bebits.read_signed_bits_unchecked(16) as i16 == b
        && bebits.read_signed_bits_unchecked(32) as i32 == c
        && bebits.read_signed_bits_unchecked(64) as i64 == d
}

#[quickcheck]
fn read_be_signed_bits2(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_be_bytes()));
    data.extend_from_slice(&(b.to_be_bytes()));
    data.extend_from_slice(&(c.to_be_bytes()));
    data.extend_from_slice(&(d.to_be_bytes()));
    let mut bebits = BigEndianReader::new(data.as_slice());
    let mut bebits2 = BigEndianReader::new(data.as_slice());

    bebits.read_signed_bits(8).map(|x| x as i8) == bebits2.read_i8()
        && bebits.read_signed_bits(16).map(|x| x as i16) == bebits2.read_i16()
        && bebits.read_signed_bits(32).map(|x| x as i32) == bebits2.read_i32()
        && bebits.read_signed_bits(64).map(|x| x as i64) == bebits2.read_i64()
}

#[quickcheck]
fn read_be_signed_unchecked_bits2(a: i8, b: i16, c: i32, d: i64) -> bool {
    let mut data = Vec::new();
    data.extend_from_slice(&(a.to_be_bytes()));
    data.extend_from_slice(&(b.to_be_bytes()));
    data.extend_from_slice(&(c.to_be_bytes()));
    data.extend_from_slice(&(d.to_be_bytes()));
    let mut bebits = BigEndianReader::new(data.as_slice());
    let mut bebits2 = BigEndianReader::new(data.as_slice());

    bebits.read_signed_bits_unchecked(8) as i8 == bebits2.read_i8_unchecked()
        && bebits.read_signed_bits_unchecked(16) as i16 == bebits2.read_i16_unchecked()
        && bebits.read_signed_bits_unchecked(32) as i32 == bebits2.read_i32_unchecked()
        && bebits.read_signed_bits_unchecked(64) as i64 == bebits2.read_i64_unchecked()
}
