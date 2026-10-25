"""
RAG Pipeline Module
Connects retrieval (Lessons 4-7) with LLM generation (Lessons 1-3).

This is the core of your Adaptive Exam Prep Engine:
1. Student asks a question
2. Retriever finds relevant chunks (with optional adaptive filtering)
3. Prompt template injects chunks + conversation history
4. LLM generates a grounded answer
5. Answer is returned with source citations

Supports: Google Gemini (free) and OpenAI GPT (paid)
"""

import os
import sys
from retriever import Retriever
from vector_store import VectorStore
from dotenv import load_dotenv
load_dotenv()

# DBMS notes contain symbols like "⋈" that crash print() on the Windows cp1252 console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# SYSTEM PROMPT TEMPLATE
# Combines: Role + Constraints + Few-shot + Negative prompting
# (All patterns from Lesson 3)
# ============================================================

SYSTEM_PROMPT = """You are a study assistant for an undergraduate/graduate level course. Your job is to help students understand concepts deeply, not just give them answers.

CONTEXT FROM COURSE MATERIAL:
{context}

RULES:
1. Answer ONLY using the provided context above. Every claim must be traceable to the context.
2. If the context does not contain enough information to answer fully, say: "Based on your course material, I can tell you that [what you found]. For the rest, you may want to check [suggest topic/chapter]."
3. When explaining, use clear language with examples. Break down complex concepts step by step.
4. Always cite which source your answer comes from, e.g., "According to [Source 1: filename | section]..."
5. If the student's question looks like a direct exam/assignment question, provide hints and guide their thinking rather than giving the direct answer.
6. Adjust your explanation depth based on the student's level: {student_level}
   - beginner: use analogies, avoid jargon, explain prerequisites
   - intermediate: balanced explanation with some technical depth
   - advanced: concise, technical, focus on edge cases and deeper insights

DO NOT:
- Make up information not present in the context
- Use phrases like "As an AI language model..."
- Give overly long explanations when a concise answer suffices
- Assume information beyond what the context provides"""


# ============================================================
# LLM API WRAPPER
# Abstracts away the difference between Gemini and OpenAI
# ============================================================

class LLMClient:
    """Wrapper around LLM APIs. Supports Gemini and OpenAI."""

    def __init__(self, provider="gemini", model=None, api_key=None):
        """
        Args:
            provider: "gemini" or "openai"
            model: Model name (defaults to a sensible cheap option)
            api_key: API key (or set GEMINI_API_KEY / OPENAI_API_KEY env var)
        """
        self.provider = provider.lower()

        if self.provider == "gemini":
            self.model_name = model or "gemini-3.6-flash"
            self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Set GEMINI_API_KEY environment variable or pass api_key.\n"
                    "Get a free key at: https://aistudio.google.com/apikey"
                )
            self._init_gemini()

        elif self.provider == "openai":
            self.model_name = model or "gpt-4o-mini"
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Set OPENAI_API_KEY environment variable or pass api_key.\n"
                    "Get a key at: https://platform.openai.com/api-keys"
                )
            self._init_openai()

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'.")

        print(f"LLM ready: {self.provider} / {self.model_name}")

    def _init_gemini(self):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)

    def _init_openai(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, system_prompt, messages, temperature=0.2):
        """
        Generate a response from the LLM.

        Args:
            system_prompt: The system instruction (with context injected)
            messages: Conversation history as list of {"role": ..., "content": ...}
            temperature: Low for RAG (stay faithful to context)

        Returns:
            The generated text response
        """
        if self.provider == "gemini":
            return self._generate_gemini(system_prompt, messages, temperature)
        else:
            return self._generate_openai(system_prompt, messages, temperature)

    def _generate_gemini(self, system_prompt, messages, temperature):
        import google.generativeai as genai

        # Gemini uses a different message format
        # Combine system prompt with first user message
        gemini_history = []

        for msg in messages[:-1]:  # all except the last message
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = self.client.start_chat(history=gemini_history)

        # Prepend system prompt to the last user message
        last_message = messages[-1]["content"]
        full_prompt = f"{system_prompt}\n\nStudent's question: {last_message}"

        response = chat.send_message(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=2500,
            ),
            request_options={"timeout": 120},  # don't hang forever on a stuck request
        )

        return response.text

    def _generate_openai(self, system_prompt, messages, temperature):
        # OpenAI has native system message support
        openai_messages = [{"role": "system", "content": system_prompt}]
        openai_messages.extend(messages)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=2500,
        )

        return response.choices[0].message.content


# ============================================================
# RAG PIPELINE
# The main class that ties everything together
# ============================================================

class RAGPipeline:
    """
    Complete RAG pipeline: retrieval + generation.

    Usage:
        rag = RAGPipeline(provider="gemini")
        answer = rag.ask("What is normalization in DBMS?")
        print(answer)
    """

    def __init__(self, provider="gemini", api_key=None, model=None,
                 collection_name="course_material"):
        """
        Initialize the full RAG pipeline.

        Args:
            provider: "gemini" or "openai"
            api_key: API key (or use environment variables)
            model: LLM model name (optional, uses sensible defaults)
            collection_name: ChromaDB collection to search
        """
        print("Initializing RAG Pipeline...")
        print("-" * 40)

        # Initialize retrieval components (from Lesson 7)
        self.store = VectorStore(collection_name=collection_name)
        self.retriever = Retriever(self.store)

        # Initialize LLM
        self.llm = LLMClient(provider=provider, model=model, api_key=api_key)

        # Conversation state
        self.conversation_history = []
        self.student_level = "intermediate"  # default

        # Config
        self.max_history_turns = 6  # keep last 6 exchanges (sliding window)
        self.num_chunks = 5         # retrieve top 5 chunks
        self.temperature = 0.2      # low temp for faithfulness
        # Rerank and adaptive modes: chunks the cross-encoder scores below this are dropped.
        # On the eval set every answerable question's top chunk scored >= 0.79 and every
        # out-of-scope question's top chunk scored < 0.001, so 0.05 sits in a wide gap.
        # If nothing passes, the question is refused without calling the LLM.
        self.min_rerank_score = 0.05

        print("-" * 40)
        print(f"Pipeline ready! Index has {self.store.collection.count()} chunks.")
        print(f"Student level: {self.student_level}")
        print()

    def set_student_level(self, level):
        """Set the student's assessed level for adaptive retrieval."""
        if level in ("beginner", "intermediate", "advanced"):
            self.student_level = level
            print(f"Student level set to: {level}")
        else:
            print("Levels: beginner, intermediate, advanced")

    def clear_history(self):
        """Reset conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def retrieve(self, question, topic=None, mode="adaptive"):
        """Run the retrieval step for the given mode and return the chunks."""
        if mode == "adaptive":
            results = self.retriever.retrieve_adaptive(
                question,
                student_level=self.student_level,
                topic=topic,
                k=self.num_chunks,
            )
            # Adaptive is built on the cross-encoder, so the same relevance guard applies
            return [r for r in results if r["score"] >= self.min_rerank_score]
        elif mode == "context":
            return self.retriever.retrieve_with_context(
                question, k=min(self.num_chunks, 3)
            )
        elif mode == "rerank":
            results = self.retriever.retrieve_with_reranking(question, k=self.num_chunks)
            return [r for r in results if r["score"] >= self.min_rerank_score]
        return self.retriever.retrieve(question, k=self.num_chunks)

    def ask(self, question, topic=None, mode="adaptive"):
        """
        Ask a question and get a RAG-grounded answer.

        Args:
            question: The student's question
            topic: Optional topic filter for retrieval
            mode: "basic", "context", "adaptive" or "rerank"

        Returns:
            The generated answer string
        """
        return self.ask_with_sources(question, topic=topic, mode=mode)["answer"]

    def ask_with_sources(self, question, topic=None, mode="adaptive", use_history=True):
        """
        Same as ask(), but also returns what was retrieved.
        Used by the evaluation (faithfulness needs the context) and the UI (sources panel).

        Args:
            use_history: If False, the question is answered without (and does not
                         update) conversation history. Evaluation uses this so
                         test cases don't leak into each other.

        Returns:
            Dict with "answer", "results" (retrieved chunks) and "context" (prompt text)
        """
        # ---- Step 1: Retrieve relevant chunks ----
        results = self.retrieve(question, topic=topic, mode=mode)

        # ---- Step 2: Format context for the prompt ----
        context = self.retriever.format_context(results)

        if not context.strip():
            answer = ("I couldn't find any relevant information in your course material "
                      "for this question. Could you rephrase it or ask about a specific topic?")
            return {"answer": answer, "results": results, "context": context}

        # ---- Step 3: Build the system prompt ----
        system_prompt = SYSTEM_PROMPT.format(
            context=context,
            student_level=self.student_level,
        )

        # ---- Step 4: Build messages with conversation history ----
        messages = list(self.conversation_history) if use_history else []
        messages.append({"role": "user", "content": question})

        # ---- Step 5: Generate answer ----
        try:
            answer = self.llm.generate(
                system_prompt=system_prompt,
                messages=messages,
                temperature=self.temperature,
            )
        except Exception as e:
            answer = f"Error generating response: {e}"
            return {"answer": answer, "results": results, "context": context}

        # ---- Step 6: Update conversation history ----
        if use_history:
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer})

            # Trim history if too long (sliding window - Lesson 2)
            if len(self.conversation_history) > self.max_history_turns * 2:
                self.conversation_history = self.conversation_history[-(self.max_history_turns * 2):]

        # ---- Step 7: Return answer with metadata ----
        return {"answer": answer, "results": results, "context": context}

    def ask_verbose(self, question, topic=None, mode="adaptive"):
        """
        Same as ask() but also prints retrieval details.
        Useful for debugging and understanding what's happening.
        """
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print(f"Mode: {mode} | Level: {self.student_level}")
        print(f"{'=' * 60}")

        # Retrieve (same path as ask(), so "context" mode works here too)
        results = self.retrieve(question, topic=topic, mode=mode)

        # Show retrieval results
        print(f"\nRetrieved {len(results)} chunks:")
        for i, r in enumerate(results):
            print(f"  [{i+1}] Score: {r['score']:.3f} | "
                  f"{r['metadata']['source']} | {r['metadata']['section']}")

        # Generate
        context = self.retriever.format_context(results)
        system_prompt = SYSTEM_PROMPT.format(
            context=context, student_level=self.student_level,
        )

        messages = list(self.conversation_history)
        messages.append({"role": "user", "content": question})

        print(f"\nContext size: {len(context)} chars (~{len(context)//4} tokens)")
        print(f"Generating answer...\n")

        try:
            answer = self.llm.generate(system_prompt, messages, self.temperature)
        except Exception as e:
            print(f"ERROR: {e}")
            return f"Error: {e}"

        # Update history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_turns * 2):]

        print(f"{'- ' * 30}")
        print(f"Answer:\n{answer}")
        print(f"{'=' * 60}\n")

        return answer


# ============================================================
# INTERACTIVE SESSION
# ============================================================

def interactive_session(provider="gemini", api_key=None, model=None):
    """Run an interactive RAG-powered study session."""

    rag = RAGPipeline(provider=provider, api_key=api_key, model=model)

    print("\nStudy Assistant Ready!")
    print("Commands: 'quit', 'level <beginner/intermediate/advanced>',")
    print("          'clear' (reset history), 'verbose' (toggle details)")
    print()

    verbose = False

    while True:
        try:
            question = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! Good luck with your studies!")
            break

        if not question:
            continue

        lower = question.lower()

        if lower in ("quit", "exit"):
            print("Bye! Good luck with your studies!")
            break

        elif lower.startswith("level "):
            level = lower.split(" ", 1)[1].strip()
            rag.set_student_level(level)
            continue

        elif lower == "clear":
            rag.clear_history()
            continue

        elif lower == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'on' if verbose else 'off'}")
            continue

        # Ask the question
        if verbose:
            answer = rag.ask_verbose(question)
        else:
            answer = rag.ask(question)
            print(f"\nAssistant > {answer}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG-powered Study Assistant")
    parser.add_argument(
        "--provider", choices=["gemini", "openai"], default="gemini",
        help="LLM provider (default: gemini, free)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (or set GEMINI_API_KEY / OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Model name (optional, uses sensible defaults)"
    )

    args = parser.parse_args()
    interactive_session(provider=args.provider, api_key=args.api_key, model=args.model)