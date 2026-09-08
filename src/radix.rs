use std::collections::{HashMap, HashSet};
use std::time::Instant;

pub type WorkerId = usize;
pub type TokenId = u32;

/// A node in the compressed Radix Tree representing a contiguous slice of token IDs.
#[derive(Debug, Clone)]
pub struct RadixNode {
    /// The compressed sequence of tokens on this edge/node.
    pub tokens: Vec<TokenId>,
    /// Set of worker IDs that currently hold this prefix in their KV cache.
    pub workers: HashSet<WorkerId>,
    /// Child nodes keyed by the FIRST token of the child's token slice.
    pub children: HashMap<TokenId, RadixNode>,
    /// Last access timestamp for LRU cache eviction.
    pub last_accessed: Instant,
}

impl RadixNode {
    pub fn new(tokens: Vec<TokenId>, worker_id: WorkerId) -> Self {
        let mut workers = HashSet::new();
        workers.insert(worker_id);
        Self {
            tokens,
            workers,
            children: HashMap::new(),
            last_accessed: Instant::now(),
        }
    }
}

/// The In-Memory Radix Tree managing KV-cache affinity across distributed LLM workers.
#[derive(Debug, Clone)]
pub struct RadixTree {
    /// Root node of the tree. The root itself contains an empty token slice.
    pub root: RadixNode,
    /// Total number of prefix nodes currently stored.
    pub node_count: usize,
}

/// Result of a Longest Prefix Match lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// The worker that has the highest prefix match for this prompt.
    pub worker_id: WorkerId,
    /// The number of contiguous tokens matched from index 0.
    pub matched_tokens: usize,
    /// The ratio of matched tokens to total prompt tokens (0.0 to 1.0).
    pub match_ratio: String,
}

impl RadixTree {
    pub fn new() -> Self {
        Self {
            root: RadixNode {
                tokens: Vec::new(),
                workers: HashSet::new(),
                children: HashMap::new(),
                last_accessed: Instant::now(),
            },
            node_count: 0,
        }
    }

    /// Insert a token sequence computed by `worker_id` into the Radix Tree.
    /// Handles branch splitting when a partial common prefix is encountered.
    pub fn insert(&mut self, tokens: &[TokenId], worker_id: WorkerId) {
        if tokens.is_empty() {
            return;
        }
        Self::insert_node(&mut self.root, tokens, worker_id, &mut self.node_count);
    }

    fn insert_node(
        current: &mut RadixNode,
        remaining: &[TokenId],
        worker_id: WorkerId,
        node_count: &mut usize,
    ) {
        current.last_accessed = Instant::now();
        current.workers.insert(worker_id);

        if remaining.is_empty() {
            return;
        }

        let first_token = remaining[0];

        if let Some(child) = current.children.get_mut(&first_token) {
            // Find length of common prefix between child.tokens and remaining
            let common_len = child
                .tokens
                .iter()
                .zip(remaining.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == child.tokens.len() {
                // Entire child edge matches; continue down the tree with remaining tokens
                Self::insert_node(child, &remaining[common_len..], worker_id, node_count);
            } else {
                // Partial match: We must split child into:
                // 1. A new split node holding the common prefix [0..common_len]
                // 2. The existing child truncated to [common_len..] (only for its original workers)
                // 3. A new child for the remaining new tokens (only for worker_id)
                let split_tokens = child.tokens[..common_len].to_vec();
                let existing_child_remaining = child.tokens[common_len..].to_vec();
                let existing_first_token = existing_child_remaining[0];

                // Truncated child preserves the original workers (who actually computed it)
                let truncated_child = RadixNode {
                    tokens: existing_child_remaining,
                    workers: child.workers.clone(),
                    children: std::mem::take(&mut child.children),
                    last_accessed: child.last_accessed,
                };

                // The split parent node now contains the common prefix and is shared by both workers
                child.tokens = split_tokens;
                child.workers.insert(worker_id);
                child.last_accessed = Instant::now();
                child.children.clear();
                child.children.insert(existing_first_token, truncated_child);

                *node_count += 1;

                if common_len < remaining.len() {
                    // There are additional new tokens; attach as a sibling child for this worker
                    let new_branch_tokens = remaining[common_len..].to_vec();
                    let new_branch_first = new_branch_tokens[0];
                    let new_child = RadixNode::new(new_branch_tokens, worker_id);
                    child.children.insert(new_branch_first, new_child);
                    *node_count += 1;
                }
            }
        } else {
            // No existing branch starts with this token; create a new leaf child
            let new_node = RadixNode::new(remaining.to_vec(), worker_id);
            current.children.insert(first_token, new_node);
            *node_count += 1;
        }
    }

    /// Perform a Longest Prefix Match (LPM) on incoming prompt tokens.
    /// Returns the worker holding the longest cached prefix and the number of tokens matched.
    pub fn find_longest_prefix(&self, tokens: &[TokenId]) -> Option<MatchResult> {
        if tokens.is_empty() {
            return None;
        }

        // Map of WorkerId -> Matched Token Count
        let mut worker_matches: HashMap<WorkerId, usize> = HashMap::new();
        self.search_recursive(&self.root, tokens, 0, &mut worker_matches);

        if worker_matches.is_empty() {
            return None;
        }

        // Find the worker with the maximum matched tokens (break ties with smaller WorkerId)
        let (&best_worker, &best_count) = worker_matches
            .iter()
            .max_by(|(w1, c1), (w2, c2)| c1.cmp(c2).then_with(|| w2.cmp(w1)))?;

        if best_count == 0 {
            return None;
        }

        let ratio = (best_count as f64) / (tokens.len() as f64);
        Some(MatchResult {
            worker_id: best_worker,
            matched_tokens: best_count,
            match_ratio: format!("{:.1}%", ratio * 100.0),
        })
    }

    fn search_recursive(
        &self,
        current: &RadixNode,
        tokens: &[TokenId],
        current_matched: usize,
        worker_matches: &mut HashMap<WorkerId, usize>,
    ) {
        if tokens.is_empty() {
            return;
        }

        let first_token = tokens[0];
        if let Some(child) = current.children.get(&first_token) {
            let common_len = child
                .tokens
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count();

            if common_len == 0 {
                return;
            }

            let new_matched = current_matched + common_len;

            // Credit all workers on this branch with the matched depth
            for &w in &child.workers {
                let entry = worker_matches.entry(w).or_insert(0);
                if new_matched > *entry {
                    *entry = new_matched;
                }
            }

            // If we matched the entire child token slice, continue deeper into children
            if common_len == child.tokens.len() && common_len < tokens.len() {
                self.search_recursive(child, &tokens[common_len..], new_matched, worker_matches);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_insert_and_exact_match() {
        let mut tree = RadixTree::new();
        let prompt = vec![101, 2054, 2003, 1037, 3231]; // 5 tokens
        tree.insert(&prompt, 1);

        let result = tree.find_longest_prefix(&prompt).expect("Should find match");
        assert_eq!(result.worker_id, 1);
        assert_eq!(result.matched_tokens, 5);
        assert_eq!(result.match_ratio, "100.0%");
    }

    #[test]
    fn test_partial_prefix_match() {
        let mut tree = RadixTree::new();
        let cached_prompt = vec![101, 2054, 2003, 1037, 3231]; // 5 tokens
        tree.insert(&cached_prompt, 1);

        // New prompt shares first 3 tokens, then diverges
        let new_prompt = vec![101, 2054, 2003, 9999, 8888];
        let result = tree.find_longest_prefix(&new_prompt).expect("Should find partial match");
        assert_eq!(result.worker_id, 1);
        assert_eq!(result.matched_tokens, 3);
        assert_eq!(result.match_ratio, "60.0%");
    }

    #[test]
    fn test_branch_splitting_across_workers() {
        let mut tree = RadixTree::new();
        // Worker 1 caches: [1, 2, 3, 4]
        tree.insert(&[1, 2, 3, 4], 1);

        // Worker 2 caches: [1, 2, 5, 6] (diverges at token 3)
        tree.insert(&[1, 2, 5, 6], 2);

        // Query with Worker 1's branch: [1, 2, 3, 99] -> should match 3 tokens on Worker 1
        let res1 = tree.find_longest_prefix(&[1, 2, 3, 99]).unwrap();
        assert_eq!(res1.worker_id, 1);
        assert_eq!(res1.matched_tokens, 3);

        // Query with Worker 2's branch: [1, 2, 5, 100] -> should match 3 tokens on Worker 2
        let res2 = tree.find_longest_prefix(&[1, 2, 5, 100]).unwrap();
        assert_eq!(res2.worker_id, 2);
        assert_eq!(res2.matched_tokens, 3);

        // Query with only common prefix: [1, 2, 888] -> both workers match 2 tokens, ties broken deterministically
        let res_common = tree.find_longest_prefix(&[1, 2, 888]).unwrap();
        assert_eq!(res_common.matched_tokens, 2);
    }

    #[test]
    fn test_no_match() {
        let mut tree = RadixTree::new();
        tree.insert(&[10, 20, 30], 1);

        // Completely different starting token
        let result = tree.find_longest_prefix(&[99, 20, 30]);
        assert!(result.is_none());
    }
}
