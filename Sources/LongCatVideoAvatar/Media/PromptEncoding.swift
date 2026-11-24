//
//  PromptEncoding.swift
//
//  Prompt → umT5-XXL features the DiT consumes, as PT `_get_t5_prompt_embeds`:
//  clean the text, tokenize padded to 512 with an attention mask, encode.
//

import Foundation
import MLX
import Tokenizers

public enum PromptEncoding {
    public static let maxSequenceLength = 512
    private static let padTokenID = 0

    /// PT `prompt_clean`: ftfy's width fix (NFKC reproduces it on well-formed
    /// input), two rounds of HTML unescaping, whitespace collapsed.
    public static func clean(_ text: String) -> String {
        var cleaned = text.precomposedStringWithCompatibilityMapping
        for _ in 0..<2 {
            cleaned = CFXMLCreateStringByUnescapingEntities(nil, cleaned as CFString, nil) as String
        }
        cleaned = cleaned.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Token ids and attention mask, both `[1, 512]`.
    public static func tokenize(_ prompt: String, tokenizer: any Tokenizer) -> (ids: MLXArray, mask: MLXArray) {
        var ids = tokenizer.encode(text: clean(prompt))
        if ids.count > maxSequenceLength {
            // HF truncation keeps the trailing </s>.
            ids = Array(ids.prefix(maxSequenceLength - 1)) + [ids.last!]
        }
        let pad = maxSequenceLength - ids.count
        let padded = ids + Array(repeating: padTokenID, count: pad)
        let mask = Array(repeating: Int32(1), count: ids.count) + Array(repeating: Int32(0), count: pad)
        return (MLXArray(padded.map(Int32.init), [1, maxSequenceLength]), MLXArray(mask, [1, maxSequenceLength]))
    }

    /// `(embeds [1, 1, 512, 4096], mask [1, 512])` for one prompt.
    public static func encode(
        _ prompt: String, encoder: UMT5EncoderModel, tokenizer: any Tokenizer
    ) -> (embeds: MLXArray, mask: MLXArray) {
        let (ids, mask) = tokenize(prompt, tokenizer: tokenizer)
        let hidden = encoder(ids, mask: mask)   // [1, 512, 4096]
        let embeds = hidden.expandedDimensions(axis: 1)
        MLX.eval(embeds)
        return (embeds, mask)
    }
}
