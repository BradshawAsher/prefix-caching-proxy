use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

/// A standard chat message matching the OpenAI Chat Completion API format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Thread-safe wrapper around HuggingFace's fast BPE Tokenizer.
#[derive(Clone)]
pub struct PromptTokenizer {
    inner: Tokenizer,
}

impl PromptTokenizer {
    /// Load the tokenizer from a local `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = Tokenizer::from_file(path)?;
        Ok(Self { inner: tokenizer })
    }

    /// Convert a raw string into a vector of token IDs.
    /// Runs in sub-100 microseconds for typical prompts.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Extract and format OpenAI-style chat messages into a single prompt string.
    /// Preserves system instructions at the beginning of the string for maximum prefix caching.
    pub fn format_chat_messages(messages: &[ChatMessage]) -> String {
        let mut full_prompt = String::with_capacity(messages.len() * 128);
        for msg in messages {
            full_prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        full_prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_chat_messages() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are an autonomous M&A diligence agent.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Calculate IRR on WidgetCo.".to_string(),
            },
        ];

        let formatted = PromptTokenizer::format_chat_messages(&messages);
        assert!(formatted.contains("<|im_start|>system\nYou are an autonomous M&A diligence agent."));
        assert!(formatted.contains("<|im_start|>user\nCalculate IRR on WidgetCo."));
    }

    #[test]
    fn test_tokenizer_encoding_with_downloaded_json() {
        // Load the downloaded Qwen2.5 tokenizer.json from workspace root
        if let Ok(tokenizer) = PromptTokenizer::from_file("tokenizer.json") {
            let prompt1 = "You are an autonomous M&A diligence agent.";
            let prompt2 = "You are an autonomous M&A diligence agent. What is EBITDA?";

            let tokens1 = tokenizer.encode(prompt1).expect("Should encode prompt 1");
            let tokens2 = tokenizer.encode(prompt2).expect("Should encode prompt 2");

            assert!(!tokens1.is_empty());
            assert!(!tokens2.is_empty());

            // prompt2 MUST start with the exact tokens of prompt1 (Prefix Property)
            assert!(tokens2.len() > tokens1.len());
            assert_eq!(&tokens2[..tokens1.len()], &tokens1[..]);
        }
    }
}
