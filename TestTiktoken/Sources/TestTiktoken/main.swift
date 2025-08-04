import Foundation
import TiktokenSwift

print("🧪 Testing TiktokenSwift...")
print("=" * 50)

do {
    // Create a test encoder
    let encoder = try TiktokenHelper.createTestEncoder()
    print("✅ Successfully created encoder")
    
    // Test encoding
    let text = "hello world!"
    let tokens = encoder.encodeText(text)
    print("\n📝 Original text: '\(text)'")
    print("🔢 Encoded tokens: \(tokens)")
    
    // Test decoding
    if let decoded = encoder.decodeTokens(tokens) {
        print("📖 Decoded text: '\(decoded)'")
        print("✅ Decoding successful!")
    } else {
        print("❌ Failed to decode tokens")
    }
    
    // Test special tokens
    let specialTokens = encoder.specialTokens()
    print("\n🎯 Special tokens: \(specialTokens)")
    
    // Test vocabulary info
    let vocabSize = encoder.nVocab()
    let maxToken = encoder.maxTokenValue()
    print("📊 Vocabulary size: \(vocabSize)")
    print("📊 Max token value: \(maxToken)")
    
    // Test encoding with details
    let details = encoder.encodeWithDetails(text: text, allowedSpecial: [])
    print("\n🔍 Encoding details:")
    print("   Tokens: \(details.tokens)")
    print("   Last piece token length: \(details.lastPieceTokenLen)")
    
    print("\n✅ All tests passed!")
    
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
