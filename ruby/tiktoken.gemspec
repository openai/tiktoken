require_relative "lib/tiktoken/version"

Gem::Specification.new do |spec|
  spec.name          = "tiktoken"
  spec.version       = Tiktoken::VERSION
  spec.summary       = "Wrapper for OpenAI's tiktoken library"
  spec.homepage      = "https://github.com/volition-co/tiktoken"
  spec.license       = "MIT"

  spec.author        = "Arjun Singh"
  spec.email         = "arjun@volition.co"

  spec.files         = Dir["*.{md,txt}", "{ext,lib}/**/*", "Cargo.*"]
  spec.require_path  = "lib"
  spec.extensions    = ["ext/tiktoken/extconf.rb"]

  spec.required_ruby_version = ">= 2.7"

  spec.add_dependency "rb_sys", "~> 0.9"
end