# ⏳ tiktoken

[English](README.md) | [简体中文](README.zh.md)

`tiktoken` 是专用于 OpenAI 模型的高速 [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding)（Byte Pair Encoding，字节对编码）分词器。

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# 获取 OpenAI API 中特定模型所对应的分词器：
enc = tiktoken.encoding_for_model("gpt-4o")
```

`tiktoken` 的开源版本可直接从 [PyPI](https://pypi.org/project/tiktoken) 安装：
```
pip install tiktoken
```

分词器 API 文档位于 `tiktoken/core.py`。

使用 `tiktoken` 的示例代码可参阅 [OpenAI Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)。


## 性能表现

`tiktoken` 比同类开源分词器快 3 到 6 倍：

![性能对比图](https://raw.githubusercontent.com/openai/tiktoken/main/perf.svg)

性能测试基于 1GB 文本，使用 GPT-2 分词器，对比环境为 `tokenizers==0.13.2` 中的 `GPT2TokenizerFast`、`transformers==4.24.0` 以及 `tiktoken==0.2.0`。


## 获取帮助

如有任何疑问，请在 [Issue 追踪器](https://github.com/openai/tiktoken/issues) 中提出。

如果您就职于 OpenAI，请查阅内部文档或随时联系 @shantanu。

## 什么是 BPE？

语言模型不像人类那样直观地阅读文本，而是接收一系列数字（称为 Token）。
字节对编码（Byte Pair Encoding, BPE）是一种将文本转换为 Token 的方式。它具有几项理想的特性：
1) **可逆且无损**：可以将 Token 完美还原为原始文本
2) **支持任意文本**：即使文本不在分词器的训练数据中也同样适用
3) **文本压缩效果**：Token 序列通常短于原始文本对应的字节数。在实践中，每个 Token 平均对应约 4 个字节。
4) **使模型能够学习常见子词**：例如，"ing" 是英语中的常见子词，因此 BPE 编码通常会将 "encoding" 拆分为形如 "encod" 和 "ing" 的 Token（而不是 "enc" 和 "oding"）。由于模型会在不同的上下文中反复看到 "ing" 这个 Token，这有助于模型更好地泛化并理解语法。

`tiktoken` 包含一个对初学者更友好的教学子模块，便于您深入了解 BPE 的底层细节，其中包含有助于可视化 BPE 过程的代码：
```python
from tiktoken._educational import *

# 在少量文本上训练一个 BPE 分词器
enc = train_simple_encoding()

# 可视化 GPT-4 编码器如何对文本进行编码
enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
enc.encode("hello world aaaaaaaaaaaa")
```


## 扩展 tiktoken

您可能希望扩展 `tiktoken` 以支持新的编码方式。有两种实现途径：


**直接按需创建 `Encoding` 对象并直接传递使用。**

```python
cl100k_base = tiktoken.get_encoding("cl100k_base")

# 在生产环境中，建议直接加载参数而非访问私有属性
# 针对特定编码的参数示例请参见 openai_public.py
enc = tiktoken.Encoding(
    # 如果修改了特殊 Token 集合，请务必使用不同的名称
    # 名称应当清晰表达所预期的行为
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)
```

**利用 `tiktoken_ext` 插件机制将自定义 `Encoding` 对象注册到 `tiktoken` 中。**

仅当您需要通过 `tiktoken.get_encoding` 检索到您的编码时才需要使用此方式，否则建议优先选择方式一。

为此，您需要在 `tiktoken_ext` 下创建一个命名空间包（namespace package）。

请按照如下结构组织项目目录，注意**不要**包含 `tiktoken_ext/__init__.py` 文件：
```
my_tiktoken_extension
├── tiktoken_ext
│   └── my_encodings.py
└── setup.py
```

`my_encodings.py` 应当是一个包含名为 `ENCODING_CONSTRUCTORS` 变量的模块。这是一个从编码名称映射到无参函数的字典，该函数返回可传递给 `tiktoken.Encoding` 以构建该编码的参数。示例请参考 `tiktoken_ext/openai_public.py`。精确细节请参见 `tiktoken/registry.py`。

您的 `setup.py` 大致如下所示：
```python
from setuptools import setup, find_namespace_packages

setup(
    name="my_tiktoken_extension",
    packages=find_namespace_packages(include=['tiktoken_ext*']),
    install_requires=["tiktoken"],
    ...
)
```

随后只需运行 `pip install ./my_tiktoken_extension`，即可使用您的自定义编码！请注意**不要**使用可编辑安装（editable install）。

---

> **免责声明**：本文档由社区志愿者进行本地化翻译，旨在促进跨语言的技术交流与学习。如翻译内容与[英文原版](README.md)存在出入，请以英文原版为准。
