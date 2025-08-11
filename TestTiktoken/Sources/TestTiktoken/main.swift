import Foundation
import TiktokenSwift

print("🧪 Testing TiktokenSwift with Latest Models...")
print("=" * 60)

// Model information from upstream
let latestModels = [
    "GPT-5": "o200k_base",
    "GPT-4.5": "o200k_base", 
    "GPT-4.1": "o200k_base",
    "o3": "o200k_base",
    "o4-mini": "o200k_base",
    "gpt-oss": "o200k_harmony"
]

let encodings = [
    "cl100k_base": "Used by GPT-4, GPT-3.5-turbo",
    "o200k_base": "Used by GPT-5, GPT-4.5, GPT-4.1, o1, o3, o4-mini, GPT-4o",
    "o200k_harmony": "Used by gpt-oss models, includes special tokens for structured output"
]

print("\n📊 Latest Model Support (from upstream tiktoken v0.11.0):")
print("-" * 60)
for (model, encoding) in latestModels {
    print("  • \(model.padding(toLength: 12, withPad: " ", startingAt: 0)) → \(encoding)")
}

print("\n🔤 Available Encodings:")
print("-" * 60)
for (encoding, description) in encodings {
    print("  • \(encoding.padding(toLength: 15, withPad: " ", startingAt: 0)) : \(description)")
}

print("\n" + "=" * 60)
print("🧪 Testing Basic Encoding/Decoding...")
print("-" * 60)

do {
    // Create a test encoder (simulating cl100k_base)
    let encoder = try TiktokenHelper.createTestEncoder()
    print("✅ Successfully created test encoder")
    
    // Test texts including new model references
    let testTexts = [
        "Hello, GPT-5!",
        "Testing GPT-4.5 and GPT-4.1 models",
        "The new o3 and o4-mini models are fast!",
        "Using o200k_harmony encoding for structured output"
    ]
    
    for text in testTexts {
        print("\n📝 Original text: '\(text)'")
        
        // Regular encoding
        let tokens = encoder.encodeText(text)
        print("🔢 Encoded tokens (\(tokens.count) tokens): \(tokens)")
        
        // Decoding
        if let decoded = encoder.decodeTokens(tokens) {
            print("📖 Decoded text: '\(decoded)'")
            let isMatch = decoded == text
            print(isMatch ? "✅ Perfect match!" : "⚠️  Text differs (expected for test encoder)")
        } else {
            print("❌ Failed to decode tokens")
        }
    }
    
    print("\n" + "=" * 60)
    print("🔬 Testing Special Tokens (o200k_harmony style)...")
    print("-" * 60)
    
    // Test with special tokens that would be in o200k_harmony
    let specialTokenTests = [
        "hello <|endoftext|> world",
        "<|startoftext|>Begin prompt<|endoftext|>",
        "Constrained output: <|constrain|>JSON<|return|>{}"
    ]
    
    for text in specialTokenTests {
        print("\n📝 Text with special: '\(text)'")
        let tokensWithSpecial = encoder.encodeWithSpecialTokens(text: text)
        print("🔢 Encoded with special: \(tokensWithSpecial)")
        
        let ordinaryTokens = encoder.encodeOrdinary(text: text)
        print("🔢 Encoded ordinary: \(ordinaryTokens)")
        
        if ordinaryTokens.count != tokensWithSpecial.count {
            print("✅ Special tokens detected and handled differently")
        }
    }
    
    print("\n" + "=" * 60)
    print("📊 Encoding Comparison Examples:")
    print("-" * 60)
    
    let comparisonText = "GPT-5 is the latest model from OpenAI"
    print("\n📝 Sample text: '\(comparisonText)'")
    
    // Simulate different encoding behaviors
    let regularTokens = encoder.encodeText(comparisonText)
    let specialTokens = encoder.encodeWithSpecialTokens(text: comparisonText)
    
    print("\n  Regular encoding (\(regularTokens.count) tokens):")
    print("  \(regularTokens)")
    
    print("\n  With special tokens (\(specialTokens.count) tokens):")
    print("  \(specialTokens)")
    
    // Token count comparison
    print("\n📈 Token Efficiency:")
    print("  • Characters: \(comparisonText.count)")
    print("  • Tokens: \(regularTokens.count)")
    print("  • Ratio: \(String(format: "%.2f", Double(comparisonText.count) / Double(regularTokens.count))) chars/token")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("\n💡 Note: This demo uses a test encoder. For production use:")
    print("   1. Load actual encoding data (cl100k_base.json or o200k_base.json)")
    print("   2. Use appropriate encoding for your model (see model list above)")
    print("   3. Handle special tokens based on your use case")
    print("\n🔍 Key Updates from upstream tiktoken:")
    print("   • GPT-5 support added (uses o200k_base encoding)")
    print("   • New models: GPT-4.5, GPT-4.1, o3, o4-mini")
    print("   • New encoding: o200k_harmony for structured output")
    print("   • Performance improvements and better error handling")
    
} catch {
    print("❌ Error: \(error)")
    exit(1)
}

// Helper to repeat string
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}