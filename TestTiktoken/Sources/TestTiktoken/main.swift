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
    
    // Test encoding with special tokens
    let textWithSpecial = "hello <|endoftext|> world"
    let tokensWithSpecial = encoder.encodeWithSpecialTokens(text: textWithSpecial)
    print("\n📝 Text with special: '\(textWithSpecial)'")
    print("🔢 Encoded tokens: \(tokensWithSpecial)")
    
    // Test ordinary encoding (without special tokens)
    let ordinaryTokens = encoder.encodeOrdinary(text: text)
    print("\n📝 Ordinary encoding: \(ordinaryTokens)")
    
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
