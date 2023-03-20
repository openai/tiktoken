begin
  require_relative "tiktoken/#{RUBY_VERSION.to_f}/tiktoken"
rescue LoadError
  require_relative "tiktoken/tiktoken"
end

require_relative "tiktoken/version"
require_relative "tiktoken/encoder"

module Tiktoken
  def self.get_encoding(encoding, extra_special_tokens={})
    Tiktoken._get_encoding(encoding, extra_special_tokens)
  end

  def self.encoding_for_model(model, extra_special_tokens={})
    Tiktoken._encoding_for_model(model, extra_special_tokens)
  end
end