# Purpose: Streaming client for the Ooba API.
# Can provide the response by token or by sentence.
#

import asyncio
from copy import deepcopy
from typing import AsyncIterator, Optional
from urllib.parse import urljoin

import aiohttp

from oobabot.sentence_splitter import SentenceSplitter


class OobaClientError(Exception):
    pass


class OobaClient:
    # Purpose: Streaming client for the Ooba API.
    # Can provide the response by token or by sentence.

    STREAMING_URI_PATH = "/api/v1/stream"
    END_OF_INPUT = ""
    DEFAULT_REQUEST_PARAMS = {
        "max_new_tokens": 250,
        "do_sample": True,
        "temperature": 1.3,
        "top_p": 0.1,
        "typical_p": 1,
        "repetition_penalty": 1.18,
        "top_k": 40,
        "min_length": 0,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }

    def __init__(self, base_url: str):
        # connector = aiohttp.TCPConnector(limit_per_host=1)

        self.api_url = urljoin(base_url, self.STREAMING_URI_PATH)
        self.total_response_tokens = 0
        self.session = aiohttp.ClientSession()

    def __del__(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.session.close())
        loop.run_until_complete(asyncio.sleep(0))  # allows aiohttp to clean up

    async def try_connect(self):
        """
        Attempt to connect to the oobabooga server.

        Returns:
            nothing, if the connection test was successful

        Raises:
            OobaClientError, if the connection fails
        """
        try:
            async with self.session.ws_connect(self.api_url) as websocket:
                return await websocket.close()

        except (ConnectionRefusedError, TimeoutError, aiohttp.WebSocketError) as e:
            raise OobaClientError(f"Failed to connect to {self.api_url}: {e}", e)

    async def request_by_sentence(self, prompt: str) -> AsyncIterator[str]:
        """
        Yields each complete sentence of the response as it arrives.
        """

        splitter = SentenceSplitter()
        async for new_token in self.request_by_token(prompt):
            for sentence in splitter.by_sentence(new_token):
                yield sentence

    async def request_by_token(self, prompt: str) -> AsyncIterator[str]:
        """
        Yields each token of the response as it arrives.
        """
        request = deepcopy(self.DEFAULT_REQUEST_PARAMS)
        request["prompt"] = prompt

        async with self.session.ws_connect(self.api_url) as websocket:
            await websocket.send_json(request)
            async for payload in websocket.receive_json():
                match payload["event"]:
                    case "text_stream":
                        if payload["text"]:
                            self.total_response_tokens += 1
                            yield payload["text"]
                    case "stream_end":
                        yield self.END_OF_INPUT
                        return
                    case _:
                        raise OobaClientError(f"Unexpected event: {payload}")
