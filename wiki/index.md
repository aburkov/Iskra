# Welcome to Iskra

Iskra is a JavaScript wiki engine that runs entirely in the browser. The wiki articles can be stored in any public GitHub repository. They are fetched dynamically when a user visits a corresponding wiki page.

Iskra comes with a search engine which builds an index and stores it on the user's browser the first time the user uses it. After this, the index is incrementally updated if new articles are added to the wiki. This means that for the search to work fast, the wiki should not be very large. If the wiki is large, the search can be deactivated in a config file.

## Introduction

Iskra is a simple wiki system that uses Markdown files for content. Here's an example of an article [Policy gradient methods in reinforcement learning](policy-gradient.md).

## Features

- **Markdown Support:** Easily create content using Markdown.
- **LaTeX Math Support:** Write inline math, e.g. $E = mc^2$, or display equations:

$$
\begin{align}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \\
\nabla \cdot \mathbf{B} &= 0 \\
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0\left(\mathbf{J} + \varepsilon_0\frac{\partial \mathbf{E}}{\partial t}\right)
\end{align}
$$

- **Section Navigation:** Jump directly to sections within your articles.
- **GitHub Integration:** Seamlessly fetch and update content from GitHub.

### Image Embedding

Embed images with custom styling:

![Decoder Block Diagram](https://github.com/user-attachments/assets/e2746e96-0e60-4946-8106-e44bfea8b806){ width=70% align=center }

### Code Snippet

Include syntax-highlighted code blocks:

```python
from collections import defaultdict

def initialize_vocabulary(corpus):
    vocabulary = defaultdict(int)
    charset = set()
    for word in corpus:
        tokenized_word = ' '.join('_' + word)
        vocabulary[tokenized_word] += 1
        charset.update(tokenized_word.split())
    return vocabulary, charset
```

### Advanced Math Example

Display more complex mathematical expressions:

$$
E = \sqrt{mc^2}\frac{1}{3}
$$

### Data Tables

Present data in structured tables:

| Model           | Num Blocks | Embedding Dim | Num Heads | Vocab Size |
|-----------------|-----------:|--------------:|----------:|-----------:|
| Our model       | 2          | 128           | 8         | 32,011     |
| Llama 3.1 8B    | 32         | 4,096         | 32        | 128,000    |
| Gemma 2 9B      | 42         | 3,584         | 16        | 256,128    |
| Gemma 2 27B     | 46         | 4,608         | 32        | 256,128    |
| Llama 3.1 70B   | 80         | 8,192         | 64        | 128,000    |
| Llama 3.1 405B  | 126        | 16,384        | 128       | 128,000    |

## Usage

Add your Markdown files to the GitHub repository and they'll be available via the wiki. The integrated search engine builds an index on your browser at first use and updates it incrementally when new articles are added.

Happy wiki-ing with Iskra!
