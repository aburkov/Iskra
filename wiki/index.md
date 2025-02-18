# Welcome to Iskra

Iskra is a JavaScript wiki engine that runs entirely in the browser. The wiki articles can be stored in any public GitHub repository. They are fetched dynamically when a user visits a corresponding wiki page.

Iskra comes with a search engine which builds an index and stores it on the user's browser the first time the user uses it. After this, the index is incrementally updated if new articles are added to the wiki. This means that for the search to work fast, the wiki should not be very large. If the wiki is large, the search can be deactivated in a config file.

## Introduction

MDWiki is a simple wiki system that uses Markdown files for content. Here's an article about [transformer](transformer.md) and [GRPO](GRPO.md).

## Features

- Markdown support
- LaTeX math support: $E = mc^2$
- Section navigation
- GitHub integration
- GitHub integration 2
- GitHub integration 3

$$
E = mc^2
$$

- Markdown support
- LaTeX math support: $E = mc^2$
- Section navigation
- GitHub integration
- GitHub integration 2

$$
E = mc^2
$$

![decoder-block-3-MoE drawio](https://github.com/user-attachments/assets/e2746e96-0e60-4946-8106-e44bfea8b806){ width=70% align=center }

- Markdown support
- LaTeX math support: $E = mc^2$
- Section navigation
- GitHub integration
- GitHub integration 2

$$
E = mc^2
$$

```python
from collections import defaultdict

def initialize_vocabulary(corpus):
    vocabulary = defaultdict(int)
    charset = set()
    for word in corpus:
        word_with_marker = '_' + word
        characters = list(word_with_marker)
        charset.update(characters)
        tokenized_word = ' '.join(characters)
        vocabulary[tokenized_word] += 1
    return vocabulary, charset
```

- Markdown support
- LaTeX math support: $E = mc^2$
- Section navigation
- GitHub integration
- GitHub integration 2

$$
E = \sqrt{mc^2}\frac{1}{3}
$$

| Doc | Text                                | Class ID | Class Name |
|-----|-------------------------------------|-------|------------|
| 1   | Movies are fun for everyone.        | 1     | Cinema     |
| 2   | Watching movies is great fun.       | 1     | Cinema     |
| 3   | Enjoy a great movie today.          | 1     | Cinema     |
| 4   | Research is interesting and important. | 3   | Science    |
| 5   | Learning math is very important.    | 3     | Science    |
| 6   | Science discovery is interesting.   | 3     | Science    |
| 7   | Rock is great to listen to.         | 2     | Music      |
| 8   | Listen to music for fun.            | 2     | Music      |
| 9   | Music is fun for everyone.          | 2     | Music      |
| 10  | Listen to folk music!               | 2     | Music      |

- Markdown support
- LaTeX math support: $E = mc^2$
- Section navigation
- GitHub integration
- GitHub integration 2

$$
E = mc^2
$$

| | `num_blocks` | `emb_dim` | `num_heads` | `vocab_size` |
|---------|---:|---:|---:|---:|
| Our model | 2 | 128 | 8 | 32,011 |
| Llama 3.1 8B | 32 | 4,096 | 32 | 128,000 |
| Gemma 2 9B | 42 | 3,584 | 16 | 256,128 |
| Gemma 2 27B | 46 | 4,608 | 32 | 256,128 |
| Llama 3.1 70B | 80 | 8,192 | 64 | 128,000 |
| Llama 3.1 405B | 126 | 16,384 | 128 | 128,000 |

- Markdown support
- LaTeX math support: $E = mc^2$
- Section navigation
- GitHub integration
- GitHub integration 2

$$
E = mc^2
$$

## Usage

Simply add your markdown files to the GitHub repository and they'll be available through the wiki.

Simply add your markdown files to the GitHub repository and they'll be available through the wiki.

Simply add your markdown files to the GitHub repository and they'll be available through the wiki.

Simply add your markdown files to the GitHub repository and they'll be available through the wiki.

Simply add your markdown files to the GitHub repository and they'll be available through the wiki.

Simply add your markdown files to the GitHub repository and they'll be available through the wiki.


Simply add your markdown files to the GitHub repository and they'll be available through the wiki.


