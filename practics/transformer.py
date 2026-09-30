"""
Core Concepts of Transformers - script version of transformer.ipynb.

Run the whole file:   python transformer.py   (each plot window pauses the script until you close it)
Or cell by cell:      VS Code shows "Run Cell" above every  # %%  line.
"""


# %% [markdown]
# Core Concepts of Transformers
#
# A Transformer is the design behind ChatGPT, Claude, BERT, Llama and almost every modern language
# model. Think of it as a reading room: every word in a sentence gets to look at every other word at
# the same time and decide which ones matter for understanding it — instead of reading left-to-right
# one word at a time like an RNN.
#
# The architecture is made of a handful of interconnected pieces. Each piece has one job, and
# together they turn a sequence of words into a representation (a list of numbers per word that
# captures its meaning in context), which can then be used to make predictions.
#
# How data flows through a Transformer (we build the pieces in exactly this order):
#  "the cat sat"                       ← raw text
#       ↓  tokenizer                   ← words → ID numbers        [12, 45, 92]
#  1. Embeddings                       ← ID numbers → meaning vectors
#  2. + Positional Encoding            ← stamp each vector with its seat number
#       ↓
#  ┌─ Transformer block (repeated N times) ───────────────┐
#  │ 3–5. (Masked) Multi-Head Self-Attention              │  ← words look at each other
#  │ 6.   Add & Norm                                      │  ← keep the original + steady the numbers
#  │ 7.   Feed-Forward Network                            │  ← each word "thinks" on its own
#  │ 6.   Add & Norm                                      │
#  └──────────────────────────────────────────────────────┘
#       ↓
#  9. Linear layer + Softmax           ← scores → probabilities for every word in the vocabulary
#       ↓
#  "on"                                ← predicted next word
#
# Where this notebook fits: this is the one-page map of all the pieces. For the deep dive (proving
# attention matches PyTorch, why √d matters, a transformer that learns to sort) see
# pytorch/phase8.ipynb. To see the pieces trained for real: tinyGpt.ipynb (decoder-only, writes
# Shakespeare) and sentimentTransformer.ipynb (encoder-only BERT, reads movie reviews).


# %%
# ==========================================
# 0. SETUP
# ==========================================
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)   # Fix the dice rolls so every run prints the same numbers
np.set_printoptions(precision=3, suppress=True)   # Print numbers to 3 decimals, no scientific notation
torch.set_printoptions(precision=3, sci_mode=False)


# %% [markdown]
# Part 1 — The Building Blocks
#
# 1. Embeddings — turning words into numbers
#
# A neural network can only do maths on numbers, so first every word (token) is swapped for an ID
# from a fixed vocabulary (the model's dictionary), and then each ID is swapped for a vector — a
# list of numbers that describes the word, like GPS coordinates for its meaning.
#
# nn.Embedding is simply a big lookup table: one row per word in the vocabulary. The rows start out
# random and are learned during training, so words used in similar ways ("cat", "dog") drift close
# together.


# %%
# ==========================================
# 1. EMBEDDINGS (word ID → meaning vector)
# ==========================================
vocabulary_size = 10000      # How many different tokens the model knows
embedding_dimension = 128    # How many numbers describe each token (GPT-2 uses 768)

# Step A: The lookup table - 10,000 rows, each a 128-number vector (random until trained)
embedding = nn.Embedding(vocabulary_size, embedding_dimension)

# Step B: A batch of 2 sentences, each already turned into 4 token IDs by a tokenizer
tokens = torch.tensor([
    [12, 45, 92, 17],
    [31, 11, 76, 12]      # Note: token 12 appears in both sentences
])

# Step C: Look up the vector for every token
vectors = embedding(tokens)

print("Token IDs shape :", tokens.shape)    # (sentences, tokens per sentence)
print("Vectors shape   :", vectors.shape)   # (sentences, tokens, numbers per token)

# The same token always gets the same vector - no matter which sentence or position it's in
print("Token 12 gets the same vector in both sentences:", torch.equal(vectors[0, 0], vectors[1, 3]))


# %% [markdown]
# That last line is a problem in disguise: "dog bites man" and "man bites dog" give the model
# exactly the same three vectors, just shuffled. Nothing yet says where each word sits. That's the
# job of the next piece.


# %% [markdown]
# 2. Positional Encoding — giving every word a seat number
#
# Self-attention (next section) compares every word with every other word, but the comparison
# ignores order — to attention, a sentence is a bag of words. So before attention runs, we add a
# unique "position fingerprint" to each word's vector: position 0 gets one pattern, position 1
# another, and so on.
#
# The original Transformer paper built these fingerprints out of sine and cosine waves of different
# speeds — like a clock with many hands: the fast hands tell neighbouring positions apart, the slow
# hands tell far-apart positions apart. Every position ends up with a unique combination.


# %%
# ==========================================
# 2. POSITIONAL ENCODING (sine/cosine "seat numbers")
# ==========================================
def positional_encoding(sequence_length, dimension):
    # Step A: A column of positions 0, 1, 2, ... (one row per seat)
    position = torch.arange(sequence_length).unsqueeze(1)
    # Step B: One "wave speed" per pair of columns - early columns wiggle fast, later ones slowly
    div_term = torch.exp(
        torch.arange(0, dimension, 2) *
        (-torch.log(torch.tensor(10000.0)) / dimension)
    )
    encoding = torch.zeros(sequence_length, dimension)
    # Step C: Even columns get sine waves, odd columns get cosine waves
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    return encoding

encoding = positional_encoding(10, 128)
print("Positional encoding shape:", encoding.shape)   # (positions, numbers per position)

# Step D: Add it to the word vectors - now the same word at a different seat is a different input
x = vectors + encoding[:4]         # (2, 4, 128) + (4, 128) → broadcast to every sentence
print("Embeddings + positions   :", x.shape)
print("Token 12 still identical in both sentences?", torch.equal(x[0, 0], x[1, 3]))   # Seat 0 vs seat 3

# Step E: Picture it - each row is one position's fingerprint
pe = positional_encoding(50, 128)
plt.figure(figsize=(9, 3.5))
plt.imshow(pe, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")   # Blue = -1, white = 0, red = +1
plt.colorbar(label="value")
plt.xlabel("embedding column  (left = fast waves, right = slow waves)")
plt.ylabel("position in sentence")
plt.title("Sinusoidal positional encoding: every row is a unique pattern")
plt.tight_layout()
plt.show()


# %% [markdown]
# Read the picture left to right: the first columns flip colour every row or two (fast hands), the
# right-hand columns barely change over 50 positions (slow hands) — which is why they show up as
# plain stripes: sine columns sit near 0 (white), cosine columns near 1 (red). No two rows are
# identical.
#
# Modern models don't all use this exact method. GPT-2 and tinyGpt.ipynb use learned position
# embeddings (just another nn.Embedding, indexed by position); Llama and most recent LLMs use RoPE
# (rotary embeddings, which rotate the query and key vectors by an angle that depends on position).
# The goal is always the same: tell attention where each word is.


# %% [markdown]
# 3. Self-Attention — how words look at each other
#
# This is the heart of the Transformer. Picture a library visit: you walk in with a query ("I need
# something about sleepy animals"). Every book has a key (the label on its spine) and a value
# (what's actually inside). You compare your query against every key, score how well each matches,
# and walk out with a blend of the books' contents — mostly from the best matches, a little from the
# rest.
#
# In a sentence, every word does this at once: each word is a reader (query) and a book (key +
# value) for all the others. It takes four steps — score, scale, soften, blend:
#
#     Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
#
#   1. Score
#       Maths: Q · Kᵀ
#       Plain English: Dot product: how well does each query match each key?
#   2. Scale
#       Maths: ÷ √d_k
#       Plain English: Shrink the scores so softmax doesn't become winner-takes-all (d_k = numbers per key)
#   3. Soften
#       Maths: softmax
#       Plain English: Turn each row of scores into percentages that add up to 1 (the attention weights)
#   4. Blend
#       Maths: × V
#       Plain English: Each word's output = weighted average of all the values
#
# Let's do it by hand with 2 tiny words, each described by 2 numbers.


# %%
# ==========================================
# 3A. SELF-ATTENTION BY HAND (NumPy, 2 words × 2 numbers)
# ==========================================
Q = np.array([[1.0, 0.0],     # Word 1's question: "what am I looking for?"
              [0.0, 1.0]])    # Word 2's question
K = np.array([[1.0, 0.0],     # Word 1's label: "what do I contain?"
              [0.5, 1.0]])    # Word 2's label
V = np.array([[2.0, 1.0],     # Word 1's content: "what I share if picked"
              [1.0, 3.0]])    # Word 2's content

# Step 1 - SCORE: every query dotted with every key. Row = the reader, column = the book
scores = Q @ K.T
print("1. Raw scores (row i = how much word i matches each word):\n", scores)

# Step 2 - SCALE: divide by √d_k so bigger vectors don't give giant scores
d_k = K.shape[1]
scaled = scores / np.sqrt(d_k)
print("\n2. Scaled scores (÷ √2):\n", scaled)

# Step 3 - SOFTEN: softmax each row so it becomes percentages adding to 1
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))   # Subtract the max first - avoids overflow, same answer
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scaled)
print("\n3. Attention weights (each row sums to 1):\n", weights)
print("   Row sums:", weights.sum(axis=1))

# Step 4 - BLEND: each word's new vector = weighted mix of every word's value
output = weights @ V
print("\n4. Output (each word now carries context from the other):\n", output)


# %% [markdown]
# Read row 1 of the weights: word 1 spends ~59% of its attention on itself and ~41% on word 2, so
# its output [1.587, 1.825] is mostly word 1's value [2, 1] with a good dose of word 2's value [1,
# 3] mixed in. The word went in "alone" and came out aware of its neighbour — that's what
# "contextual representation" means.
#
# Where do Q, K and V come from? Above we typed them in. In a real model they are all made from the
# same input vectors x, each through its own learned linear layer (a learned "translation"). Same
# word, three roles. That's why it's called self-attention: the sentence attends to itself.


# %%
# ==========================================
# 3B. SELF-ATTENTION IN PYTORCH (Q, K, V learned from the same input)
# ==========================================
d_model = 8                     # Numbers per word (tiny, so we can print things)
sentence = torch.randn(1, 5, d_model)   # 1 sentence, 5 words, 8 numbers each (pretend embeddings)

# Step A: Three learned "translators" - one per role
to_query = nn.Linear(d_model, d_model, bias=False)   # "What am I looking for?"
to_key   = nn.Linear(d_model, d_model, bias=False)   # "What do I contain?"
to_value = nn.Linear(d_model, d_model, bias=False)   # "What do I share?"

q, k, v = to_query(sentence), to_key(sentence), to_value(sentence)

# Step B: The four steps - score, scale, soften, blend
scores  = q @ k.transpose(-2, -1) / math.sqrt(d_model)   # (1, 5, 5): every word vs every word
weights = F.softmax(scores, dim=-1)                       # Each row → percentages
out     = weights @ v                                     # (1, 5, 8): same shape as the input

print("Attention weights (5 words × 5 words):\n", weights[0].detach())   # .detach() = just the numbers, no training history
print("Output shape:", out.shape, "- same as input, but every word now carries context")

# Step C: PyTorch has the whole thing as one fast function - check it gives the same answer
out_builtin = F.scaled_dot_product_attention(q, k, v)
print("Matches F.scaled_dot_product_attention:", torch.allclose(out, out_builtin, atol=1e-6))


# %% [markdown]
# The weights are spread out fairly evenly because the linear layers are random — nothing has been
# learned yet. After training, a row like "it" in "the animal didn't cross the street because it was
# tired" concentrates its weight on "animal".
#
# The output has the same shape as the input: 5 words × 8 numbers in, 5 words × 8 numbers out.
# That's what lets us stack attention layers on top of each other.


# %% [markdown]
# 4. Masks — rules about who may look at whom
#
# Sometimes a word must not be allowed to look at certain other words. The trick is always the same:
# set the forbidden scores to −∞ before softmax. Since e^(-∞) = 0, those words get exactly 0%
# attention, and the remaining percentages still add up to 1.
#
# - Causal mask (no peeking at the future) — used by every text generator (GPT, Claude, the
#   decoder). When the model learns to predict word 4, it must only see words 1–3, otherwise it
#   would just copy the answer. The mask is a lower triangle: row i can see columns 0…i.
# - Padding mask (ignore the filler) — sentences in a batch have different lengths, so short ones
#   get padded with a dummy <pad> token to line them up. Nobody should pay attention to padding.


# %%
# ==========================================
# 4. MASKS (causal + padding)
# ==========================================
raw = scores[0].detach()   # The 5×5 scores from section 3B

# Step A: Causal mask - True where looking is ALLOWED (a lower triangle)
n = 5
causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
print("Causal mask (1 = allowed):\n", causal.int())

# Step B: Block the future by setting those scores to -infinity, then softmax
masked_scores  = raw.masked_fill(~causal, float("-inf"))
causal_weights = F.softmax(masked_scores, dim=-1)
print("\nAttention weights with causal mask:\n", causal_weights)
# Word 0 can only see itself (100%), word 1 splits between words 0-1, ... word 4 sees everyone

# Step C: Padding mask - pretend the last 2 of our 5 tokens are <pad> filler
is_real = torch.tensor([True, True, True, False, False])
pad_weights = F.softmax(raw.masked_fill(~is_real, float("-inf")), dim=-1)
print("\nAttention weights with padding mask (last 2 columns are <pad>):\n", pad_weights)

# Step D: Picture the causal pattern
plt.figure(figsize=(5, 3.8))
plt.imshow(causal_weights, cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="attention weight")
plt.xlabel("word being looked AT (key)")
plt.ylabel("word doing the looking (query)")
plt.title("Causal mask: nobody sees the future")
plt.tight_layout()
plt.show()


# %% [markdown]
# Everything above the diagonal is exactly 0 — the future is invisible. With the padding mask, the
# last two columns are 0 for every word, so the filler contributes nothing to anyone's output.
#
# In PyTorch's built-in layers you rarely build these by hand:
# nn.Transformer.generate_square_subsequent_mask(n) makes the causal mask, and key_padding_mask= /
# src_key_padding_mask= take the padding mask (careful: there, True means "ignore this position" —
# the opposite of our is_real).


# %% [markdown]
# 5. Multi-Head Attention — several readers at once
#
# One attention pattern gives each word one way of looking at the sentence. But language has many
# relationships at the same time: which noun does "it" refer to, which adjective describes which
# noun, which verb goes with which subject…
#
# Multi-head attention runs several attention "heads" side by side, like a team of readers each
# hunting for a different kind of clue. The 128 numbers per word are split between the heads (8
# heads × 16 numbers each), every head does its own score–scale–soften–blend, and the results are
# glued back together and mixed by one final linear layer. Splitting (rather than copying) means 8
# heads cost about the same as 1 big head.


# %%
# ==========================================
# 5. MULTI-HEAD ATTENTION
# ==========================================
embedding_size = 128
number_of_heads = 8
print("Numbers each head works with:", embedding_size // number_of_heads)   # 128 / 8 = 16

attention = nn.MultiheadAttention(
    embed_dim=embedding_size,
    num_heads=number_of_heads,
    batch_first=True          # Shapes are (batch, tokens, features) - the order used everywhere in this notebook
)

x = torch.randn(2, 10, embedding_size)   # 2 sentences, 10 tokens, 128 numbers each

# Self-attention = the same x plays query, key AND value
output, weights = attention(x, x, x, average_attn_weights=False)

print("Output shape :", output.shape)    # (2, 10, 128) - same as input
print("Weights shape:", weights.shape)   # (2 sentences, 8 heads, 10 queries, 10 keys) - one pattern per head
print("Parameters   :", sum(p.numel() for p in attention.parameters()))   # 4 weight matrices of 128×128 + biases


# %% [markdown]
# The weights shape shows the point: 8 separate 10×10 attention patterns per sentence, one per head.
# The parameter count is 4 × (128 × 128 + 128) = 66,048 — the Q, K, V and output projections. The
# number of heads does not change it.


# %% [markdown]
# 6. Add & Norm — residual connections and layer normalization
#
# Every sub-layer (attention, feed-forward) is wrapped in two helpers:
#
# - Add (residual connection): output = input + sub-layer(input). The sub-layer only has to learn a
#   correction to what's already there, not rebuild the word from scratch. It's also a highway for
#   gradients (the learning signal) to flow straight back through a deep stack of layers — without
#   it, 12, 24 or 96 layers wouldn't train.
# - Norm (LayerNorm): rescales each word's numbers to average 0 and spread 1, so values can't slowly
#   blow up or shrink as they pass through layer after layer. Like re-tuning an instrument between
#   songs.


# %%
# ==========================================
# 6. ADD & NORM
# ==========================================
layer_norm = nn.LayerNorm(128)

x = torch.randn(2, 10, 128) * 5 + 3   # Pretend input with "drifted" numbers: average ~3, spread ~5
attention_output = attention(x, x, x)[0]

# Step A: ADD - keep the original and add what attention found
# Step B: NORM - re-centre each word's 128 numbers
result = layer_norm(x + attention_output)

print("Shape:", result.shape)
print(f"Before norm → average {(x + attention_output).mean():.2f}, spread {(x + attention_output).std():.2f}")
print(f"After norm  → average {result.mean():.2f}, spread {result.std():.2f}")


# %% [markdown]
# Post-norm vs pre-norm. The original paper (and the code above) normalizes after adding:
# LayerNorm(x + sublayer(x)). Almost every modern model (GPT-2 onward, tinyGpt.ipynb) normalizes
# before the sub-layer instead: x + sublayer(LayerNorm(x)) — it trains more stably when stacking
# many layers. In PyTorch's built-in layers this is the norm_first=True switch.


# %% [markdown]
# 7. Position-wise Feed-Forward Network — each word thinks on its own
#
# Attention is where words talk to each other. The feed-forward network (FFN) is where each word
# thinks by itself about what it just heard. It's a small two-layer network applied to every word
# separately (that's what "position-wise" means — same network, each position on its own, no mixing
# between words).
#
# It widens each word's vector (128 → 512, the usual 4×), applies a non-linearity (ReLU; modern
# models prefer GELU or SwiGLU), then shrinks it back to 128 so the next layer gets the same shape.


# %%
# ==========================================
# 7. POSITION-WISE FEED-FORWARD NETWORK
# ==========================================
feed_forward = nn.Sequential(
    nn.Linear(128, 512),   # Widen: 128 → 512 numbers (more room to "think")
    nn.ReLU(),             # Non-linearity - lets it learn more than straight-line relationships
    nn.Linear(512, 128)    # Shrink back to 128 so the output fits the next layer
)

x = torch.randn(2, 10, 128)
ff_out = feed_forward(x)
print("Feed-forward output shape:", ff_out.shape)   # Same shape in and out

# "Position-wise" proof: running word 3 on its own gives the same answer as running the whole sentence
alone = feed_forward(x[:, 3:4, :])
print("Word 3 alone == word 3 in the sentence:", torch.allclose(alone, ff_out[:, 3:4, :], atol=1e-6))

ffn_params = sum(p.numel() for p in feed_forward.parameters())
attn_params = sum(p.numel() for p in attention.parameters())
print(f"\nFeed-forward parameters: {ffn_params:,}")
print(f"Attention parameters   : {attn_params:,}")
print(f"Share held by the FFN  : {ffn_params / (ffn_params + attn_params):.0%}")


# %% [markdown]
# The humble FFN holds about two thirds of each layer's weights. Attention decides where information
# flows; the feed-forward layers do much of the storing and transforming — a lot of what a large
# language model "knows" is thought to live in them.


# %% [markdown]
# 8. Putting It Together — one Transformer block from scratch
#
# Now we snap sections 5–7 together into the unit that gets stacked 12, 24, 96… times in real
# models:
#
# x ─┬─> Multi-Head Attention ─> (+) ─> LayerNorm ─┬─> Feed-Forward ─> (+) ─> LayerNorm ─> out
#    └─────────────────────────────┘               └─────────────────────┘
#           residual connection                        residual connection
#
# Then we check it against PyTorch's nn.TransformerEncoderLayer: if we've understood every piece,
# the two should have exactly the same number of weights.


# %%
# ==========================================
# 8. ONE TRANSFORMER BLOCK FROM SCRATCH
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=8, d_ff=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)   # Section 5
        self.norm1 = nn.LayerNorm(d_model)                                           # Section 6
        self.feed_forward = nn.Sequential(                                           # Section 7
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Step A: words look at each other, then Add & Norm
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        # Step B: each word thinks on its own, then Add & Norm
        x = self.norm2(x + self.feed_forward(x))
        return x

block = TransformerBlock()
x = torch.randn(2, 10, 128)
print("Our block output shape:", block(x).shape)

# Compare with PyTorch's built-in version of the same thing
builtin = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True)
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"Our block parameters     : {count(block):,}")
print(f"nn.TransformerEncoderLayer: {count(builtin):,}")


# %% [markdown]
# Identical counts: attention (66,048) + feed-forward (131,712) + two LayerNorms (2 × 256) =
# 198,272. There's nothing hidden inside PyTorch's layer that you haven't just written yourself (the
# built-in one adds dropout and fast fused kernels, which change speed, not the design).


# %% [markdown]
# 9. Output Layer & Softmax — from vectors to a prediction
#
# After the last block, each position's vector goes through one final linear layer that produces a
# score (logit) for every word in the vocabulary, and softmax turns those scores into probabilities.
#
# How we then pick the next word is called the decoding strategy:
# - Greedy — always take the highest-probability word. Safe but repetitive.
# - Sampling with temperature — roll a weighted die. Divide the logits by a temperature first: below
#   1 makes the model more confident/boring, above 1 more adventurous/random.
# - Top-k — only roll the die among the k most likely words, so absurd picks are impossible.


# %%
# ==========================================
# 9. OUTPUT LAYER, SOFTMAX AND DECODING
# ==========================================
vocab = ["mat", "sofa", "moon", "roof"]      # A tiny 4-word vocabulary for "the cat sat on the ..."
logits = torch.tensor([[2.1, 1.2, 0.4, 3.0]])   # Raw scores from the final linear layer

# Step A: Softmax turns scores into probabilities that add up to 1
probabilities = F.softmax(logits, dim=-1)
for word, p in zip(vocab, probabilities[0]):
    print(f"  {word:5s} {p:.1%}")
print("Sum:", probabilities.sum().item())

# Step B: Greedy - just take the top one
print("\nGreedy pick:", vocab[probabilities.argmax()])

# Step C: Temperature reshapes the odds before sampling
for temperature in [0.5, 1.0, 2.0]:
    p = F.softmax(logits / temperature, dim=-1)[0]
    print(f"Temperature {temperature}: " + "  ".join(f"{w} {q:.0%}" for w, q in zip(vocab, p)))

# Step D: Top-k sampling (k=2) - only the 2 best candidates stay in the draw
top_values, top_ids = logits.topk(2)
pick = top_ids[0, torch.multinomial(F.softmax(top_values, dim=-1)[0], 1)]
print("\nTop-2 candidates:", [vocab[i] for i in top_ids[0]], "→ sampled:", vocab[pick])


# %% [markdown]
# Low temperature (0.5) pushes almost everything onto "roof"; high temperature (2.0) flattens the
# odds so even "moon" gets a real chance. That's the "creativity" slider you see in chatbot APIs.


# %% [markdown]
# Part 2 — The Full Architecture
#
# 10. Encoder–Decoder: the original design
#
# The original Transformer ("Attention Is All You Need", 2017) was built for translation, and has
# two halves:
#
# - Encoder — reads the whole input sentence (e.g. English) with no mask: every word sees every
#   other word, past and future. Output: one context-rich vector per input word, called the memory.
# - Decoder — writes the output sentence (e.g. French) one word at a time. Each decoder layer has
#   three sub-layers, not two:
#   1. Masked self-attention — looks at the French words written so far (causal mask, no peeking).
#   2. Cross-attention — the queries come from the decoder, the keys and values come from the
#      encoder's
#      memory. This is the moment the translator glances back at the English sentence.
#   3. Feed-forward, as usual (each with Add & Norm).
#
# English: "the cat sat"                      French so far: "<start> le chat"
#         ↓                                            ↓
#    ┌─────────┐    memory (K, V)             ┌──────────────────────────┐
#    │ ENCODER │ ───────────────────────────> │ DECODER                  │
#    │ N blocks│                              │ masked self-attention    │
#    └─────────┘                              │ cross-attention ←memory  │
#                                             │ feed-forward             │
#                                             └──────────────────────────┘
#                                                      ↓
#                                             next word: "s'est"


# %%
# ==========================================
# 10A. THE ENCODER WORKFLOW
# ==========================================
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,           # Numbers per token
    nhead=8,               # Attention heads
    dim_feedforward=512,   # Width of the feed-forward "thinking" layer
    batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)   # Stack 4 copies of the block

src = torch.randn(2, 20, 128)   # 2 source sentences, 20 tokens each (embeddings + positions)

# Sentence 2 is really only 16 tokens long - the last 4 are <pad>. True = "ignore this position"
src_padding = torch.zeros(2, 20, dtype=torch.bool)
src_padding[1, 16:] = True

memory = encoder(src, src_key_padding_mask=src_padding)
print("Encoder output (memory) shape:", memory.shape)   # One contextual vector per input token


# %%
# ==========================================
# 10B. THE DECODER WORKFLOW
# ==========================================
decoder_layer = nn.TransformerDecoderLayer(
    d_model=128,
    nhead=8,
    dim_feedforward=512,
    batch_first=True
)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

target = torch.randn(2, 15, 128)   # The output sentence so far: 15 tokens (embeddings + positions)

# Causal mask: 0 where looking is allowed, -inf where it's the future
causal_mask = nn.Transformer.generate_square_subsequent_mask(15)
print("Causal mask (top-left 5×5 corner):\n", causal_mask[:5, :5])

decoded = decoder(
    target,
    memory,                                    # Cross-attention looks here (the encoder's output)
    tgt_mask=causal_mask,                      # No peeking at future output words
    memory_key_padding_mask=src_padding        # Don't glance back at the source's <pad> tokens
)
print("\nDecoder output shape:", decoded.shape)   # One vector per OUTPUT token

# Proof the mask works: changing the LAST target word must not change the output at earlier positions
target_changed = target.clone()
target_changed[:, -1] = torch.randn(128)
decoder.eval()   # Switch off dropout so both runs are comparable
with torch.no_grad():
    a = decoder(target, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_padding)
    b = decoder(target_changed, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_padding)
print("Earlier positions unaffected by a future word:", torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5))
print("Last position did change                     :", not torch.allclose(a[:, -1], b[:, -1], atol=1e-5))


# %% [markdown]
# The last check is the whole point of the causal mask: we changed word 15 and positions 1–14 didn't
# move at all. So during training we can feed the entire correct target sentence in one go (called
# teacher forcing) and every position still only learns from the words before it — 15 training
# examples in a single pass.


# %% [markdown]
# 11. A Complete (Tiny) Translation Model
#
# Sections 1–10 in one class: embeddings → positions → nn.Transformer (encoder + decoder) → output
# layer. Then the generation loop: at prediction time there's no correct answer to feed in, so the
# decoder starts with a <start> token, predicts one word, appends it, and runs again — until it
# predicts <end>.
#
# The model below is untrained, so the words it picks are random. The point is the plumbing and the
# loop, which are exactly the same in a trained translator.


# %%
# ==========================================
# 11. A COMPLETE ENCODER-DECODER MODEL
# ==========================================
class TinyTranslator(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, max_len=100):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)      # 1. Source word IDs → vectors
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)      #    Target word IDs → vectors
        self.register_buffer("pe", positional_encoding(max_len, d_model))   # 2. Seat numbers (not learned)
        self.transformer = nn.Transformer(                     # 3–8. Encoder + decoder stacks
            d_model=d_model, nhead=8, num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=512, batch_first=True
        )
        self.to_vocab = nn.Linear(d_model, tgt_vocab)          # 9. Vector → a score per target word

    def forward(self, src_ids, tgt_ids):
        src = self.src_embed(src_ids) + self.pe[: src_ids.shape[1]]
        tgt = self.tgt_embed(tgt_ids) + self.pe[: tgt_ids.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(tgt_ids.shape[1])
        out = self.transformer(src, tgt, tgt_mask=mask)
        return self.to_vocab(out)                              # Logits: (batch, target length, target vocab)

START, END = 1, 2                                               # Special token IDs
model = TinyTranslator(src_vocab=1000, tgt_vocab=1200)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training shape check: feed the whole target at once (teacher forcing)
src_ids = torch.randint(3, 1000, (2, 12))   # 2 English sentences, 12 tokens
tgt_ids = torch.randint(3, 1200, (2, 9))    # Their French translations so far, 9 tokens
print("Logits shape:", model(src_ids, tgt_ids).shape, "→ a score for every French word, at every position")

# Generation: greedy, one word at a time
model.eval()
with torch.no_grad():
    generated = torch.tensor([[START]])                  # Step A: begin with <start>
    for _ in range(6):
        logits = model(src_ids[:1], generated)           # Step B: run on everything written so far
        next_id = logits[0, -1].argmax()                 # Step C: only the LAST position predicts the next word
        generated = torch.cat([generated, next_id.view(1, 1)], dim=1)   # Step D: append and repeat
        if next_id == END:
            break
print("Generated token IDs:", generated[0].tolist(), "(random - the model is untrained)")


# %% [markdown]
# 12. Three Families of Transformers
#
# Almost every famous model keeps only one half, or both, of the original design:
#
#   Encoder-only
#       Which parts: Encoder stack
#       Attention mask: None — sees the whole text both ways
#       Good at: Understanding: classification, search, tagging
#       Famous examples: BERT, RoBERTa, DistilBERT
#       In this repo: sentimentTransformer.ipynb
#   Decoder-only
#       Which parts: Decoder stack without cross-attention
#       Attention mask: Causal
#       Good at: Generating text, next-word prediction, chat
#       Famous examples: GPT, Claude, Llama
#       In this repo: tinyGpt.ipynb
#   Encoder–decoder
#       Which parts: Both + cross-attention
#       Attention mask: None in encoder, causal in decoder
#       Good at: Turning one sequence into another: translation, summarising
#       Famous examples: Original Transformer, T5, BART
#       In this repo: Section 11 above
#
# In PyTorch terms: an encoder-only model is nn.TransformerEncoder + a small classifier head on top;
# a decoder-only model is also built from nn.TransformerEncoderLayer blocks — just with a causal
# mask passed in — because without cross-attention a "decoder" block is identical to an encoder
# block.


# %% [markdown]
# Part 3 — Limitations of Transformer Architecture
#
# 1. High computational requirements. Every word is compared with every other word, so the work
#    grows with the square of the text length: double the text, four times the attention work. (The
#    cell below puts numbers on it.)
# 2. Memory consumption. The same square shows up in memory — an n × n attention table per head, per
#    layer — plus, at generation time, the KV cache (the stored keys and values of every earlier
#    word, so they aren't recomputed). Tricks like FlashAttention avoid ever storing the full table,
#    but the amount of work is still squared.
# 3. Fixed context window. A model can only attend to so many tokens at once (its context length).
#    Anything beyond that is simply invisible to it unless extra machinery (retrieval, summarising)
#    feeds it back in.
# 4. Large training and data requirements. Attention has almost no built-in assumptions about
#    language (unlike a CNN's "nearby pixels matter" or an RNN's "order matters") — it has to learn
#    all of that from examples. That flexibility is why transformers scale so well, but it also
#    means they need huge datasets and lots of compute. On small datasets, simpler models often win
#    (compare Phase 7, where bag-of-words beat an LSTM on IMDB).
# 5. Hallucination. A language model is trained to produce the most plausible next word, not the
#    most true one. When it doesn't know, it can still write a fluent, confident, wrong answer.
# 6. Hard to interpret. Attention weights show where a model looked, but not really why it made a
#    decision. Understanding what billions of weights have learned is an open research area.


# %%
# ==========================================
# LIMITATION 1 & 2 IN NUMBERS: THE QUADRATIC COST
# ==========================================
# Memory for ONE full attention table (n × n scores, 4 bytes each) - for one head, in one layer
print(f"{'tokens':>8} | {'scores in table':>16} | {'memory (1 head, 1 layer)':>24}")
for n in [512, 2_048, 8_192, 32_768, 131_072]:
    scores_count = n * n
    megabytes = scores_count * 4 / 1024**2
    size = f"{megabytes:,.0f} MB" if megabytes < 1024 else f"{megabytes / 1024:,.0f} GB"
    print(f"{n:>8,} | {scores_count:>16,} | {size:>24}")
# Real models have dozens of heads and layers on top of this


# %% [markdown]
# Each ×4 in text length is ×16 in attention cost. At 131k tokens a single head's table would need
# 64 GB — which is why long-context models depend on memory-saving tricks like FlashAttention
# (compute the table in small tiles and never store the whole thing) rather than the plain maths
# above.
#
# Cheat Sheet — shapes through the whole pipeline
#
#   Token IDs
#       Shape: (batch, seq)
#       What it means: One integer per token
#   Embedding + positional encoding
#       Shape: (batch, seq, d_model)
#       What it means: One meaning-plus-position vector per token
#   Attention weights (per head)
#       Shape: (batch, heads, seq, seq)
#       What it means: Who looks at whom — the quadratic part
#   After each block (attention + FFN)
#       Shape: (batch, seq, d_model)
#       What it means: Same shape in and out — that's why blocks stack
#   Output logits
#       Shape: (batch, seq, vocab)
#       What it means: A score for every vocabulary word at every position
#   Softmax
#       Shape: (batch, seq, vocab)
#       What it means: Probabilities — pick with greedy / temperature / top-k
