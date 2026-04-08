import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(page_title="Fractal NLP Analyzer", layout="wide")

st.title("Fractal Analysis in Natural Language Processing")
st.markdown("""
Analyze text complexity using fractal concepts. In linguistics, fractal geometry reveals itself through **Power Laws**. 
We measure this using **Zipf's Law** (frequency distributions) and **Heaps' Law** (vocabulary scaling behavior).
""")

# -----------------------------
# Text Input
# -----------------------------
st.sidebar.header("Input & Settings")
text = st.sidebar.text_area("Paste your text corpus here:", height=300, 
                            value="The quick brown fox jumps over the lazy dog. " * 50)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return words

# -----------------------------
# 1. Zipf's Law Analysis
# -----------------------------
def zipf_analysis(words):
    freq = Counter(words)
    sorted_freq = sorted(freq.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freq) + 1)
    
    log_ranks = np.log(ranks)
    log_freqs = np.log(sorted_freq)
    
    # Use numpy.polyfit instead of scipy
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    alpha = -slope 
    
    # Calculate R-squared using numpy
    correlation_matrix = np.corrcoef(log_ranks, log_freqs)
    r_squared = correlation_matrix[0, 1] ** 2
    
    return ranks, sorted_freq, alpha, r_squared, log_ranks, intercept

# -----------------------------
# 2. Heaps' Law (NLP Scaling / Box Counting Equivalent)
# -----------------------------
def heaps_scaling_analysis(words):
    text_lengths = []
    vocab_sizes = []
    current_vocab = set()
    
    step = max(1, len(words) // 50) 
    
    for i, word in enumerate(words):
        current_vocab.add(word)
        if (i + 1) % step == 0 or i == len(words) - 1:
            text_lengths.append(i + 1)
            vocab_sizes.append(len(current_vocab))
            
    log_N = np.log(text_lengths)
    log_V = np.log(vocab_sizes)
    
    # Use numpy.polyfit instead of scipy
    slope, intercept = np.polyfit(log_N, log_V, 1)
    beta = slope
    
    # Calculate R-squared using numpy
    correlation_matrix = np.corrcoef(log_N, log_V)
    r_squared = correlation_matrix[0, 1] ** 2
    
    return text_lengths, vocab_sizes, beta, r_squared, log_N, intercept

# -----------------------------
# Main Execution
# -----------------------------
if text.strip():
    words = preprocess(text)
    
    if len(words) < 20:
        st.warning("Please enter a longer text (at least 20 words) for meaningful fractal analysis.")
    else:
        st.write(f"### Total Corpus Size: **{len(words)}** tokens | Unique Vocabulary: **{len(set(words))}** tokens")
        st.write("---")
        
        col1, col2 = st.columns(2)
        
        # Column 1: Zipf's Law
        with col1:
            st.subheader("1. Zipf's Law (Distributional Fractal)")
            ranks, freqs, alpha, r2_zipf, log_ranks, intercept_zipf = zipf_analysis(words)
            
            m1, m2 = st.columns(2)
            m1.metric("Zipf Exponent (α)", f"{alpha:.3f}", help="Standard natural language usually approaches 1.0")
            m2.metric("Fit Quality (R²)", f"{r2_zipf:.3f}", help="1.0 means perfect power-law (fractal) behavior.")
            
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.loglog(ranks, freqs, marker='.', linestyle='none', color='#1f77b4', label='Actual Data')
            fit_freqs = np.exp(intercept_zipf) * ranks ** (-alpha)
            ax1.loglog(ranks, fit_freqs, color='red', linestyle='--', label=f'Fit (α={alpha:.2f})')
            ax1.set_xlabel("Rank (Log Scale)")
            ax1.set_ylabel("Frequency (Log Scale)")
            ax1.legend()
            ax1.grid(True, which="both", ls="--", alpha=0.5)
            st.pyplot(fig1)
            
            st.info("**What does this mean?** Zipf's law states that the frequency of any word is inversely proportional to its rank. In mathematics, this is a Power Law distribution. A straight line on a log-log plot indicates scale invariance—a core property of fractals.")

        # Column 2: Heaps' Law
        with col2:
            st.subheader("2. Heaps' Law (Scaling Behavior)")
            lengths, vocabs, beta, r2_heaps, log_N, intercept_heaps = heaps_scaling_analysis(words)
            
            m3, m4 = st.columns(2)
            m3.metric("Scaling Exponent (β)", f"{beta:.3f}", help="Standard English typically ranges from 0.4 to 0.6")
            m4.metric("Fit Quality (R²)", f"{r2_heaps:.3f}", help="1.0 means perfect power-law scaling.")
            
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.loglog(lengths, vocabs, marker='.', linestyle='none', color='#2ca02c', label='Actual Growth')
            fit_vocabs = np.exp(intercept_heaps) * np.array(lengths) ** beta
            ax2.loglog(lengths, fit_vocabs, color='red', linestyle='--', label=f'Fit (β={beta:.2f})')
            ax2.set_xlabel("Text Length (Log Scale)")
            ax2.set_ylabel("Vocabulary Size (Log Scale)")
            ax2.legend()
            ax2.grid(True, which="both", ls="--", alpha=0.5)
            st.pyplot(fig2)
            
            st.info("**What does this mean?** This replaces box-counting. Heaps' law measures how vocabulary scales with text length. Just like measuring the length of a coastline depends on your 'measuring stick,' the size of your vocabulary depends on the size of your text sample following a power-law exponent.")
else:
    st.info("Enter text in the sidebar to start the analysis.")
    

give requirements.txt for this streamlit