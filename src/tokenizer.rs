use std::collections::HashMap;

// Wildandev Elis BPE Tokenizer
// Byte-level BPE: vocab dibangun dari frekuensi pair pada corpus.
pub struct WildandevTokenizer {
    vocab: Vec<Vec<u8>>,               // id -> bytes
    vocab_map: HashMap<Vec<u8>, usize>, // bytes -> id
    merges: Vec<(usize, usize)>,       // (left_id, right_id) -> new id
    merge_ranks: HashMap<(usize, usize), usize>,
}

impl WildandevTokenizer {
    pub fn train(corpus: &str, target_vocab: usize) -> Self {
        let mut vocab: Vec<Vec<u8>> = (0..256).map(|b| vec![b as u8]).collect();
        let mut vocab_map: HashMap<Vec<u8>, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        // Words as byte sequences (split by whitespace & newline, keep delimiters)
        let mut words: Vec<Vec<usize>> = Vec::new();
        for chunk in corpus.split(|c: char| c == ' ' || c == '\n') {
            let seq: Vec<usize> = chunk.bytes().map(|b| b as usize).collect();
            if !seq.is_empty() {
                words.push(seq);
            }
        }

        let mut merges = Vec::new();
        let mut merge_ranks: HashMap<(usize, usize), usize> = HashMap::new();

        while vocab.len() < target_vocab {
            // Count adjacent pairs
            let mut pair_count: HashMap<(usize, usize), usize> = HashMap::new();
            for w in &words {
                for i in 0..w.len().saturating_sub(1) {
                    *pair_count.entry((w[i], w[i + 1])).or_insert(0) += 1;
                }
            }

            // Find most frequent pair
            let best = match pair_count.iter().max_by_key(|(_, &c)| c) {
                Some((pair, &count)) if count > 1 => *pair,
                _ => break, // nothing left to merge
            };

            let new_id = vocab.len();
            let mut merged = vocab[best.0].clone();
            merged.extend_from_slice(&vocab[best.1]);
            vocab.push(merged.clone());
            vocab_map.insert(merged, new_id);
            merges.push(best);
            merge_ranks.insert(best, merges.len() - 1);

            // Apply merge across all words
            for w in words.iter_mut() {
                let mut i = 0;
                while i + 1 < w.len() {
                    if w[i] == best.0 && w[i + 1] == best.1 {
                        w[i] = new_id;
                        w.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        Self { vocab, vocab_map, merges, merge_ranks }
    }

    // Encode raw text into token IDs using learned merges
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut seq: Vec<usize> = text.bytes().map(|b| b as usize).collect();
        if seq.is_empty() {
            return seq;
        }

        loop {
            // find lowest-rank merge applicable
            let mut best_rank: Option<(usize, usize, usize)> = None; // (rank, pos, new_id)
            for i in 0..seq.len().saturating_sub(1) {
                if let Some(&rank) = self.merge_ranks.get(&(seq[i], seq[i + 1])) {
                    if best_rank.is_none() || rank < best_rank.unwrap().0 {
                        let new_id = self.merges[rank].0 * 0 + self.merge_lookup(rank);
                        best_rank = Some((rank, i, new_id));
                    }
                }
            }
            match best_rank {
                Some((_, pos, new_id)) => {
                    seq[pos] = new_id;
                    seq.remove(pos + 1);
                }
                None => break,
            }
        }
        seq
    }

    fn merge_lookup(&self, rank: usize) -> usize {
        let (a, b) = self.merges[rank];
        let mut merged = self.vocab[a].clone();
        merged.extend_from_slice(&self.vocab[b]);
        *self.vocab_map.get(&merged).unwrap()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(v) = self.vocab.get(id) {
                bytes.extend_from_slice(v);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn merges_count(&self) -> usize {
        self.merges.len()
    }

    // Save/Load binary format
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&(self.vocab.len() as u32).to_le_bytes())?;
        f.write_all(&(self.merges.len() as u32).to_le_bytes())?;
        for v in &self.vocab {
            f.write_all(&(v.len() as u32).to_le_bytes())?;
            f.write_all(v)?;
        }
        for (a, b) in &self.merges {
            f.write_all(&(*a as u32).to_le_bytes())?;
            f.write_all(&(*b as u32).to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut data = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut data)?;
        let mut pos = 0usize;

        let read_u32 = |pos: &mut usize, data: &[u8]| -> u32 {
            let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
            *pos += 4;
            v
        };

        let vocab_len = read_u32(&mut pos, &data) as usize;
        let merges_len = read_u32(&mut pos, &data) as usize;

        let mut vocab = Vec::with_capacity(vocab_len);
        let mut vocab_map = HashMap::new();
        for i in 0..vocab_len {
            let len = read_u32(&mut pos, &data) as usize;
            let bytes = data[pos..pos + len].to_vec();
            pos += len;
            vocab_map.insert(bytes.clone(), i);
            vocab.push(bytes);
        }

        let mut merges = Vec::with_capacity(merges_len);
        let mut merge_ranks = HashMap::new();
        for r in 0..merges_len {
            let a = read_u32(&mut pos, &data) as usize;
            let b = read_u32(&mut pos, &data) as usize;
            merges.push((a, b));
            merge_ranks.insert((a, b), r);
        }

        Ok(Self { vocab, vocab_map, merges, merge_ranks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let corpus = "wildandev elis engine wildandev elis wildandev";
        let tok = WildandevTokenizer::train(corpus, 300);
        let text = "wildandev elis";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_compression() {
        let corpus = "wildandev wildandev wildandev elis elis elis engine engine engine";
        let tok = WildandevTokenizer::train(corpus, 300);
        let ids = tok.encode("wildandev wildandev");
        // "wildandev wildandev" -> [token("wildandev"), space(32), token("wildandev")] = 3 tokens vs 19 raw bytes!
        assert_eq!(ids.len(), 3, "BPE compression: expected 3 tokens, got {:?}", ids);
        assert_eq!(tok.decode(&ids), "wildandev wildandev");
    }
}
