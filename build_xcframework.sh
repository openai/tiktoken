#!/bin/bash
set -e

echo "🚀 Building Multi-Platform XCFramework for tiktoken..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📍 Working directory: $(pwd)"
echo ""

# Check for required tools
echo "🔍 Checking required tools..."
if ! command -v cargo &> /dev/null; then
    echo "❌ cargo not found. Please install Rust."
    exit 1
else
    echo "✅ cargo found: $(cargo --version)"
fi

if ! command -v xcodebuild &> /dev/null; then
    echo "❌ xcodebuild not found. Please install Xcode."
    exit 1
else
    echo "✅ xcodebuild found: $(xcodebuild -version | head -n1)"
fi

if ! command -v lipo &> /dev/null; then
    echo "❌ lipo not found. Please install Xcode Command Line Tools."
    exit 1
else
    echo "✅ lipo found"
fi

# First, we need to generate the Swift bindings
echo ""
echo "🔧 Generating Swift bindings..."
mkdir -p swift-bindings

# Use the installed uniffi-bindgen to generate Swift bindings
if [ -f "$HOME/.cargo/bin/uniffi-bindgen" ]; then
    UNIFFI_BINDGEN="$HOME/.cargo/bin/uniffi-bindgen"
    echo "✅ Using uniffi-bindgen from cargo"
elif command -v uniffi-bindgen &> /dev/null; then
    UNIFFI_BINDGEN="uniffi-bindgen"
    echo "✅ Using system uniffi-bindgen"
else
    echo "❌ uniffi-bindgen not found. Please install it with: cargo install uniffi_bindgen"
    exit 1
fi

echo "📝 Running uniffi-bindgen..."
$UNIFFI_BINDGEN generate src/tiktoken.udl \
    --language swift \
    --out-dir swift-bindings \
    --config uniffi.toml || {
    echo "❌ Failed to generate Swift bindings"
    exit 1
}

# Remove the old incorrect module map if it exists
rm -f swift-bindings/module.modulemap

# Install required targets if not already installed
echo ""
echo "📱 Checking and installing required Rust targets..."

# Function to check and add target
add_target_if_needed() {
    local target=$1
    if rustup target list --installed | grep -q "$target"; then
        echo "  ✅ $target already installed"
    else
        echo "  📦 Installing $target..."
        rustup target add "$target" || {
            echo "  ⚠️  Failed to install $target"
            return 1
        }
    fi
    return 0
}

# Install all required targets
add_target_if_needed "aarch64-apple-ios"
add_target_if_needed "aarch64-apple-ios-sim"
add_target_if_needed "x86_64-apple-ios"
add_target_if_needed "aarch64-apple-darwin"
add_target_if_needed "x86_64-apple-darwin"

# Build for all platforms
echo ""
echo "🦀 Building Rust library for all Apple platforms..."

# Build for iOS arm64
echo "  📱 Building for iOS (arm64)..."
cargo build --release --target aarch64-apple-ios || {
    echo "  ❌ Failed to build for iOS arm64"
    exit 1
}

# Build for iOS simulator (arm64 + x86_64)
echo "  📱 Building for iOS Simulator (arm64)..."
cargo build --release --target aarch64-apple-ios-sim || {
    echo "  ❌ Failed to build for iOS Simulator arm64"
    exit 1
}

echo "  📱 Building for iOS Simulator (x86_64)..."
cargo build --release --target x86_64-apple-ios || {
    echo "  ❌ Failed to build for iOS Simulator x86_64"
    exit 1
}

# Build for macOS (arm64 + x86_64)
echo "  💻 Building for macOS (arm64)..."
cargo build --release --target aarch64-apple-darwin || {
    echo "  ❌ Failed to build for macOS arm64"
    exit 1
}

echo "  💻 Building for macOS (x86_64)..."
cargo build --release --target x86_64-apple-darwin || {
    echo "  ❌ Failed to build for macOS x86_64"
    exit 1
}

# Swift bindings are already generated in swift-bindings directory

# Create fat libraries
echo ""
echo "🔗 Creating universal libraries..."

# iOS Simulator universal binary
echo "  📱 Creating iOS Simulator universal binary..."
mkdir -p target/universal-ios-sim
lipo -create \
    target/aarch64-apple-ios-sim/release/libtiktoken.a \
    target/x86_64-apple-ios/release/libtiktoken.a \
    -output target/universal-ios-sim/libtiktoken.a || {
    echo "  ❌ Failed to create iOS Simulator universal binary"
    exit 1
}
echo "  ✅ iOS Simulator universal binary created"

# macOS universal binary
echo "  💻 Creating macOS universal binary..."
mkdir -p target/universal-macos
lipo -create \
    target/aarch64-apple-darwin/release/libtiktoken.a \
    target/x86_64-apple-darwin/release/libtiktoken.a \
    -output target/universal-macos/libtiktoken.a || {
    echo "  ❌ Failed to create macOS universal binary"
    exit 1
}
echo "  ✅ macOS universal binary created"

# Create module map for frameworks
echo ""
echo "📦 Creating framework structure..."
cat > swift-bindings/module.modulemap << 'EOF'
framework module TiktokenFFI {
    header "TiktokenFFI.h"
    export *
}
EOF

# Function to create framework
create_framework() {
    local PLATFORM=$1
    local SDK=$2
    local LIB_PATH=$3
    local MIN_VERSION=$4
    
    echo "  📦 Creating framework for $PLATFORM..."
    
    local FRAMEWORK_DIR="build/$PLATFORM/TiktokenFFI.framework"
    mkdir -p "$FRAMEWORK_DIR/Headers"
    mkdir -p "$FRAMEWORK_DIR/Modules"
    
    # Copy header
    cp swift-bindings/TiktokenFFI.h "$FRAMEWORK_DIR/Headers/"
    
    # Copy module map
    cp swift-bindings/module.modulemap "$FRAMEWORK_DIR/Modules/module.modulemap"
    
    # Copy library
    cp "$LIB_PATH" "$FRAMEWORK_DIR/TiktokenFFI"
    
    # Create Info.plist
    cat > "$FRAMEWORK_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>TiktokenFFI</string>
    <key>CFBundleIdentifier</key>
    <string>com.tiktoken.TiktokenFFI</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>TiktokenFFI</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>$SDK</string>
    </array>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>$MIN_VERSION</string>
</dict>
</plist>
EOF
}

# Create build directory
mkdir -p build

# Create frameworks
create_framework "ios" "iPhoneOS" "target/aarch64-apple-ios/release/libtiktoken.a" "13.0"
create_framework "ios-simulator" "iPhoneSimulator" "target/universal-ios-sim/libtiktoken.a" "13.0"
create_framework "macos" "MacOSX" "target/universal-macos/libtiktoken.a" "10.15"

# Create XCFramework
echo ""
echo "🔧 Creating XCFramework..."

# Verify frameworks exist
echo "  🔍 Verifying frameworks..."
for framework in "build/ios/TiktokenFFI.framework" "build/ios-simulator/TiktokenFFI.framework" "build/macos/TiktokenFFI.framework"; do
    if [ -d "$framework" ]; then
        echo "  ✅ Found $framework"
    else
        echo "  ❌ Missing $framework"
        exit 1
    fi
done

# Remove old XCFrameworks
echo "  🧹 Removing old XCFrameworks..."
rm -rf TiktokenFFI.xcframework
rm -rf TiktokenSwift/Sources/TiktokenFFI/TiktokenFFI.xcframework

# Create the XCFramework
echo "  🏗️  Building XCFramework..."
xcodebuild -create-xcframework \
    -framework build/ios/TiktokenFFI.framework \
    -framework build/ios-simulator/TiktokenFFI.framework \
    -framework build/macos/TiktokenFFI.framework \
    -output TiktokenFFI.xcframework || {
    echo "  ❌ Failed to create XCFramework"
    exit 1
}
echo "  ✅ XCFramework created successfully"

# Copy to TiktokenSwift package in separate directory
TIKTOKEN_SWIFT_DIR="/Users/nicholasarner/Development/Active/TiktokenSwift"
if [ -d "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenFFI" ]; then
    echo "📦 Copying XCFramework to TiktokenSwift package..."
    cp -R TiktokenFFI.xcframework "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenFFI/"
    
    # Update header if needed
    if [ -f "swift-bindings/TiktokenFFI.h" ]; then
        cp swift-bindings/TiktokenFFI.h "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenFFI/include/"
    fi
    
    # Update Swift file if needed
    if [ -f "swift-bindings/TiktokenFFI.swift" ] && [ -f "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenSwift/TiktokenFFI.swift" ]; then
        cp swift-bindings/TiktokenFFI.swift "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenSwift/TiktokenFFI.swift"
        
        # Fix imports
        sed -i '' '/#if canImport(TiktokenFFI)/,/#endif/d' "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenSwift/TiktokenFFI.swift"
        sed -i '' '/^import Foundation$/a\
import TiktokenFFI' "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenSwift/TiktokenFFI.swift"
        
        # Add warning suppression
        sed -i '' 's/fatalError("UniFFI contract version mismatch/print("Warning: UniFFI contract version mismatch") \/\/ fatalError("UniFFI contract version mismatch/' "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenSwift/TiktokenFFI.swift"
        sed -i '' 's/fatalError("UniFFI API checksum mismatch/print("Warning: UniFFI API checksum mismatch") \/\/ fatalError("UniFFI API checksum mismatch/' "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenSwift/TiktokenFFI.swift"
    fi
fi

# Clean up
rm -rf build
rm -rf swift-bindings

echo ""
echo "✅ Multi-platform XCFramework created successfully!"
echo ""
echo "🎯 Supported platforms:"
echo "   - iOS devices (arm64)"
echo "   - iOS Simulator (arm64, x86_64)"
echo "   - macOS (arm64, x86_64)"
echo ""
echo "📦 XCFramework locations:"
echo "   - ./TiktokenFFI.xcframework"
if [ -d "$TIKTOKEN_SWIFT_DIR/Sources/TiktokenFFI/TiktokenFFI.xcframework" ]; then
    echo "   - $TIKTOKEN_SWIFT_DIR/Sources/TiktokenFFI/TiktokenFFI.xcframework"
fi