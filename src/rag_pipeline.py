"""
rag_pipeline.py

Task 3: RAG core logic.

Built on the PRE-BUILT vector store supplied for Tasks 3-4:
    data/raw/complaint_embeddings.parquet
        columns: id (str), document (str), embedding (384-dim float array,
                 from all-MiniLM-L6-v2), metadata (dict: complaint_id, company,
                 product, product_category, issue, date_received, chunk_index, ...)

Since the embeddings are already computed, this module does NOT re-embed the
corpus. Instead:
    - src/build_index.py reads the parquet once, stacks all embeddings into a
      FAISS index, and saves it + the aligned text/metadata to data/faiss_index/.
    - This module's Retriever loads that FAISS index + metadata at startup,
      embeds only the user's QUESTION (using the same all-MiniLM-L6-v2 model),
      and runs FAISS similarity search to get the top-k chunks.

Usage:
    from src.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    result = rag.answer("Why are customers unhappy with credit card billing disputes?")
    print(result["answer"])
    for src in result["sources"]:
        print(src["metadata"], src["content"][:200])
"""

import logging
import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions
about customer complaints. Use the following retrieved complaint excerpts to formulate
your answer. Base your answer only on the provided context. If the context doesn't contain
enough information to answer the question, state clearly that you don't have enough
information rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Lightweight stand-in for a retrieved chunk (replaces LangChain's Document --
# we no longer depend on LangChain/Chroma for retrieval at all)
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """A single retrieved chunk: its text, metadata, and similarity score."""

    __slots__ = ("page_content", "metadata", "score")

    def __init__(self, page_content: str, metadata: Dict[str, Any], score: float):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.score = score


def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Turn a list of retrieved chunks into a single context string, tagging each
    excerpt with identifying metadata so the LLM (and a human reviewer) can
    trace claims back to a source complaint.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        tag = (
            f"[Excerpt {i} | Complaint ID: {meta.get('complaint_id', 'N/A')} | "
            f"Product: {meta.get('product_category', meta.get('product', 'N/A'))} | "
            f"Issue: {meta.get('issue', 'N/A')} | "
            f"Company: {meta.get('company', 'N/A')}]"
        )
        blocks.append(f"{tag}\n{chunk.page_content}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Loads the FAISS index + aligned metadata produced by build_index.py
    (built from the pre-supplied complaint_embeddings.parquet), and answers
    similarity-search queries against it.
    """

    def __init__(
        self,
        index_path: str = config.FAISS_INDEX_PATH,
        meta_path: str = config.FAISS_META_PATH,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
    ):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No FAISS index found at '{index_path}'. Run "
                f"`python -m src.build_index` first to build it from "
                f"data/raw/complaint_embeddings.parquet."
            )
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No metadata file found at '{meta_path}'.")

        logger.info(f"Loading embedding model '{embedding_model_name}' (for queries only)...")
        # We only ever embed the user's QUESTION with this model at query time.
        # The corpus itself was already embedded with this same model when
        # complaint_embeddings.parquet was produced -- using SentenceTransformer
        # directly here (rather than going through Chroma/LangChain) keeps
        # the query embedding identical in format to what build_index.py
        # stacked into the FAISS index.
        self.embedder = SentenceTransformer(embedding_model_name)

        logger.info(f"Loading FAISS index from '{index_path}'...")
        self.index = faiss.read_index(index_path)
        logger.info(f"Index loaded with {self.index.ntotal:,} vectors.")

        logger.info(f"Loading aligned metadata from '{meta_path}'...")
        self.meta_df = pd.read_parquet(meta_path)
        if len(self.meta_df) != self.index.ntotal:
            logger.warning(
                f"Row count mismatch: index has {self.index.ntotal:,} vectors "
                f"but metadata has {len(self.meta_df):,} rows. Results may be misaligned."
            )

    def retrieve(
        self,
        question: str,
        k: int = config.DEFAULT_TOP_K,
        product_filter: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Embed `question` with the same model used to build the index, then
        run FAISS similarity search for the top-k most similar chunks.

        product_filter is applied AFTER the FAISS search (over-fetching more
        candidates first) since the base IndexFlatIP has no native metadata
        filtering -- this is a simple post-filter, fine at this dataset scale.
        """
        query_vec = self.embedder.encode([question], normalize_embeddings=True).astype("float32")

        # Over-fetch when a filter is active, since some of the top results
        # may get filtered out and we still want k results back if possible.
        search_k = k * 10 if product_filter else k
        search_k = min(search_k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, search_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.meta_df.iloc[idx]
            meta = row["metadata"] or {}

            if product_filter:
                row_product = meta.get("product_category") or meta.get("product")
                if row_product != product_filter:
                    continue

            results.append(RetrievedChunk(page_content=row["document"], metadata=meta, score=float(score)))
            if len(results) >= k:
                break

        return results


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class Generator:
    """
    Thin wrapper so the pipeline can swap backends without touching
    retrieval/prompt code. Two backends supported out of the box:

      - "anthropic": calls the Anthropic API (needs ANTHROPIC_API_KEY env var
                      and billing set up). Best quality.
      - "local_hf":   runs Qwen2.5-1.5B-Instruct locally via transformers.
                      No API key, no cost, decent quality for a 1.5B model.
    """

    def __init__(self, backend: str = config.GENERATOR_BACKEND):
        self.backend = backend

        if backend == "anthropic":
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "ANTHROPIC_API_KEY is not set. Export it before running, "
                    "or set RAG_GENERATOR_BACKEND=local_hf to use a local model instead."
                )
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = config.ANTHROPIC_MODEL_NAME

        elif backend == "local_hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info(f"Loading local generator model '{config.LOCAL_HF_MODEL_NAME}'...")
            # Qwen2.5-Instruct is a causal LM (unlike FLAN-T5's seq2seq
            # architecture), with a much larger context window (32K tokens),
            # so the retrieved chunks won't get silently truncated the way
            # they were with FLAN-T5-base's 512-token encoder limit.
            self._local_tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_HF_MODEL_NAME)
            self._local_model = AutoModelForCausalLM.from_pretrained(config.LOCAL_HF_MODEL_NAME)

        else:
            raise ValueError(f"Unknown generator backend: {backend}")

    def generate(self, prompt: str) -> str:
        if self.backend == "anthropic":
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if block.type == "text"
            ).strip()

        elif self.backend == "local_hf":
            # Qwen2.5-Instruct expects chat-formatted input (a list of
            # role/content messages run through its chat template), not a
            # raw string, to follow instructions properly.
            #
            # apply_chat_template(..., return_tensors="pt") can return a
            # BatchEncoding wrapper (dict-like, with .input_ids inside) rather
            # than a bare tensor depending on the transformers version --
            # passing that wrapper directly into model.generate() then fails
            # because .generate() expects a plain tensor. To avoid depending
            # on which form gets returned, we instead render the template to
            # a plain string (tokenize=False) and tokenize it explicitly
            # ourselves, which always gives us a predictable plain tensor.
            messages = [{"role": "user", "content": prompt}]
            chat_text = self._local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._local_tokenizer(chat_text, return_tensors="pt")
            input_ids = inputs["input_ids"]

            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,  # deterministic, more reliable for factual QA
            )
            # Slice off the input tokens so we only decode the newly
            # generated continuation, not the echoed prompt.
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            generated = self._local_tokenizer.decode(new_tokens, skip_special_tokens=True)
            return generated.strip()

    def generate_stream(self, prompt: str):
        """
        Same as generate(), but yields the answer incrementally (token-by-token
        for local_hf, chunk-by-chunk for anthropic) instead of returning the
        full string at once. Used by the Gradio UI for streaming output.
        """
        if self.backend == "anthropic":
            with self.client.messages.stream(
                model=self.model_name,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        elif self.backend == "local_hf":
            import threading
            from transformers import TextIteratorStreamer

            messages = [{"role": "user", "content": prompt}]
            chat_text = self._local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._local_tokenizer(chat_text, return_tensors="pt")

            # TextIteratorStreamer lets generate() run on a background thread
            # while we read decoded text pieces from this thread as they're
            # produced -- this is what makes token-by-token streaming possible
            # with a local transformers model (model.generate() itself is a
            # single blocking call with no native streaming return value).
            streamer = TextIteratorStreamer(
                self._local_tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                streamer=streamer,
            )
            thread = threading.Thread(target=self._local_model.generate, kwargs=generation_kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text

            thread.join()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(
        self,
        index_path: str = config.FAISS_INDEX_PATH,
        meta_path: str = config.FAISS_META_PATH,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
        generator_backend: str = config.GENERATOR_BACKEND,
        top_k: int = config.DEFAULT_TOP_K,
    ):
        self.top_k = top_k
        self.retriever = Retriever(
            index_path=index_path, meta_path=meta_path, embedding_model_name=embedding_model_name
        )
        self.generator = Generator(backend=generator_backend)

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        product_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full retrieve -> prompt -> generate flow.

        Returns a dict with:
            - "answer": the LLM's generated answer (str)
            - "sources": list of {"content": str, "metadata": dict, "score": float}
                         for each retrieved chunk, for citation / inspection / eval tables
            - "prompt": the fully assembled prompt actually sent to the LLM
        """
        k = k or self.top_k
        chunks = self.retriever.retrieve(question, k=k, product_filter=product_filter)

        if not chunks:
            return {
                "answer": "I don't have enough information to answer that — no relevant complaints were retrieved.",
                "sources": [],
                "prompt": None,
            }

        context = format_context(chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        answer = self.generator.generate(prompt)

        sources = [
            {"content": c.page_content, "metadata": c.metadata, "score": c.score} for c in chunks
        ]
        return {"answer": answer, "sources": sources, "prompt": prompt}

    def answer_stream(
        self,
        question: str,
        k: Optional[int] = None,
        product_filter: Optional[str] = None,
    ):
        """
        Streaming counterpart to answer(). Does retrieval once (same as
        answer()), then yields (partial_answer_so_far, sources) tuples as the
        generator produces text incrementally. The UI re-renders on every
        yield, which is what gives the token-by-token "typing" effect.

        The final yielded tuple has the complete answer, identical in content
        to what answer() would have returned in "answer".
        """
        k = k or self.top_k
        chunks = self.retriever.retrieve(question, k=k, product_filter=product_filter)

        sources = [
            {"content": c.page_content, "metadata": c.metadata, "score": c.score} for c in chunks
        ]

        if not chunks:
            yield (
                "I don't have enough information to answer that — no relevant complaints were retrieved.",
                sources,
            )
            return

        context = format_context(chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        partial = ""
        for piece in self.generator.generate_stream(prompt):
            partial += piece
            yield partial, sources


if __name__ == "__main__":
    rag = RAGPipeline()
    test_questions = [
        "Why are customers unhappy with credit card billing disputes?",
        "What problems do people report with money transfers?",
    ]
    for q in test_questions:
        result = rag.answer(q)
        print(f"\nQ: {q}\nA: {result['answer']}\n")
        for s in result["sources"][:2]:
            print(f"  - [{s['metadata'].get('complaint_id')}] {s['content'][:120]}...")