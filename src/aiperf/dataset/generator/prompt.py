# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiperf.common.config import PromptConfig
from aiperf.common.exceptions import (
    ConfigurationError,
    InvalidStateError,
    NotInitializedError,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset import utils
from aiperf.dataset.generator.base import BaseGenerator

DEFAULT_CORPUS_FILE = "assets/shakespeare.txt"


class PromptGenerator(BaseGenerator):
    """A class for generating synthetic prompts from a text corpus.

    This class loads a text corpus (e.g., Shakespearean text), tokenizes it,
    and uses the tokenized corpus to generate synthetic prompts of specified
    lengths. It supports generating prompts with a target number of tokens
    (with optional randomization around a mean and standard deviation) and
    can reuse previously generated token blocks to optimize generation for
    certain use cases. It also allows for the creation of a pool of prefix
    prompts that can be randomly selected.
    """

    def __init__(self, config: PromptConfig, tokenizer: Tokenizer, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self._tokenized_corpus = None
        self._corpus_size = 0
        self._prefix_prompts: list[str] = []
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        # Cached prompts: block ID -> list of tokens
        self._cache: dict[int, list[int]] = {}

        # TODO: move this under initialize() method
        # Initialize corpus if not already done
        if self._tokenized_corpus is None:
            self._initialize_corpus()

        # Initialize prefix prompts pool if the pool size > 0
        if self.config.prefix_prompt.pool_size > 0:
            self._create_prefix_prompt_pool()

    def _initialize_corpus(self) -> None:
        """Load and tokenize the corpus once, storing it for reuse."""
        corpus_path = Path(__file__).parent / DEFAULT_CORPUS_FILE

        with open(corpus_path) as f:
            lines = f.readlines()

        def tokenize_chunk(chunk):
            cleaned_text = " ".join(line.strip() for line in chunk if line.strip())
            tokens = self.tokenizer.encode(cleaned_text)
            return tokens

        num_threads = os.cpu_count()
        if num_threads is None:
            num_threads = 4

        # Ensure chunk_size is at least 1 to avoid division by zero in range()
        chunk_size = max(1, len(lines) // num_threads)
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            tokenized_chunks = list(executor.map(tokenize_chunk, chunks))

        self._tokenized_corpus = [
            token for chunk in tokenized_chunks for token in chunk
        ]
        self._corpus_size = len(self._tokenized_corpus)
        self.debug(lambda: f"Initialized corpus with {self._corpus_size} tokens")

    def _create_prefix_prompt_pool(self) -> None:
        """Generate a pool of prefix prompts to sample from."""
        if self._tokenized_corpus is None:
            raise NotInitializedError("Tokenized corpus is not initialized.")

        self._prefix_prompts = [
            self._generate_prompt(self.config.prefix_prompt.length)
            for _ in range(self.config.prefix_prompt.pool_size)
        ]
        self.debug(
            lambda: f"Initialized prefix prompts pool with {len(self._prefix_prompts)} prompts"
        )

    def generate(
        self,
        mean: int | None = None,
        stddev: int | None = None,
        hash_ids: list[int] | None = None,
    ) -> str:
        """Generate a synthetic prompt with the configuration parameters.

        Args:
            mean: The mean of the normal distribution.
            stddev: The standard deviation of the normal distribution.
            hash_ids: A list of hash indices used for token reuse.

        Returns:
            A synthetic prompt as a string.
        """
        if hash_ids:
            return self._generate_cached_prompt(
                mean, hash_ids, self.config.input_tokens.block_size
            )

        num_tokens = utils.sample_positive_normal_integer(mean, stddev)
        return self._generate_prompt(num_tokens)

    def _generate_prompt(self, num_tokens: int) -> str:
        """Generate a prompt containing exactly `num_tokens` number of tokens.

        Args:
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A synthetic prompt as a string.
        """
        return self.tokenizer.decode(self._sample_tokens(num_tokens))

    def _generate_cached_prompt(
        self,
        num_tokens: int,
        hash_ids: list[int],
        block_size: int,
    ) -> str:
        """
        Generate a prompt containing exactly `num_tokens` by reusing previously generated prompts
        stored in `_cache`. Each hash index in `hash_ids` corresponds to a block of
        `block_size` tokens. If a hash index is found in `_cache`, its stored prompt is reused.
        Otherwise, a new prompt is generated using `_generate_prompt()` and stored in `_cache`.

        Args:
            num_tokens: The number of tokens required in the prompt.
            hash_ids: A list of hash IDs to use for token reuse.
            block_size: The number of tokens allocated per hash block.

        Returns:
            str: A synthetic prompt as a string.

        Raises:
            ConfigurationError: If the input parameters are not compatible.
        """
        final_prompt: list[int] = []
        current_block_size = block_size

        # Sanity check the final block size
        final_block_size = num_tokens - ((len(hash_ids) - 1) * block_size)
        if final_block_size <= 0 or block_size < final_block_size:
            raise ConfigurationError(
                f"Input length: {num_tokens}, Hash IDs: {hash_ids}, Block size: {block_size} "
                f"are not compatible. The final hash block size: {final_block_size} must be "
                f"greater than 0 and less than or equal to {block_size}."
            )

        for index, hash_id in enumerate(hash_ids):
            # For the last hash ID, use the remaining tokens as the block size
            if index == len(hash_ids) - 1:
                current_block_size = final_block_size

            if hash_id not in self._cache:
                # To ensure that the prompt doesn't merge chunks, we insert a BOS or EOS token
                # at the beginning. Length is maintained and the prompt generates the expected
                # number of tokens. If no BOS or EOS token is available, we don't insert one.
                prompt_tokens: list[int] = []
                if self.tokenizer.block_separation_token_id is not None:
                    prompt_tokens += [self.tokenizer.block_separation_token_id]
                    prompt_tokens += self._sample_tokens(current_block_size - 1)
                else:
                    prompt_tokens += self._sample_tokens(current_block_size)

                self._cache[hash_id] = prompt_tokens  # store to cache

            final_prompt.extend(self._cache[hash_id])

        return self.tokenizer.decode(final_prompt, skip_special_tokens=False)

    def _sample_tokens(self, num_tokens: int) -> list[int]:
        """Generate a list of token IDs containing exactly `num_tokens` number of tokens
        using the preloaded tokenized corpus.

        Args:
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A list of token IDs.

        Raises:
            NotInitializedError: If the tokenized corpus is not initialized
        """
        if not self._tokenized_corpus:
            raise NotInitializedError("Tokenized corpus is not initialized.")
        if num_tokens > self._corpus_size:
            self.warning(
                f"Requested prompt length {num_tokens} is longer than the corpus. "
                f"Returning a prompt of length {self._corpus_size}."
            )

        start_idx = random.randrange(self._corpus_size)

        end_idx = start_idx + num_tokens
        prompt_tokens = self._tokenized_corpus[start_idx:end_idx]
        if end_idx > self._corpus_size:
            prompt_tokens += self._tokenized_corpus[: end_idx - self._corpus_size]

        self.trace(lambda: f"Sampled {len(prompt_tokens)} tokens from corpus")
        return prompt_tokens

    def get_random_prefix_prompt(self) -> str:
        """
        Fetch a random prefix prompt from the pool.

        Returns:
            A random prefix prompt.

        Raises:
            InvalidStateError: If the prefix prompts pool is empty.
        """
        if not self._prefix_prompts:
            raise InvalidStateError(
                "Attempted to sample a prefix prompt but the prefix prompts pool is empty. "
                "Please ensure that the prefix prompts pool is initialized."
            )
        return random.choice(self._prefix_prompts)
