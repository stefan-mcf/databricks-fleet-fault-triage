"""Versioned project runbooks and real Search keyword/vector/hybrid adapters."""

import json
import re
import time
from pathlib import Path

import numpy as np

from fleet_fault_triage.contracts import identity


class Corpus:
    def __init__(self, documents):
        if not documents or len(documents) > 200:
            raise ValueError("corpus must contain 1–200 approved chunks")
        self.documents = {d["id"]: d for d in documents}
        if sum(len(d["content"].split()) for d in documents) > 50_000:
            raise ValueError("corpus exceeds the frozen word allowance")
        if len(self.documents) != len(documents):
            raise ValueError("duplicate chunk ID")
        for document in documents:
            if set(document) != {"id", "title", "content", "check", "sha256"}:
                raise ValueError("unexpected corpus fields")
            if document["sha256"] != identity({k: v for k, v in document.items() if k != "sha256"}):
                raise ValueError("corpus chunk hash changed")
        self.version = identity(sorted(documents, key=lambda d: d["id"]))

    @classmethod
    def load(cls, root: Path):
        records = []
        for path in sorted(root.glob("*.json")):
            record = json.loads(path.read_text())
            record["sha256"] = identity(record)
            records.append(record)
        return cls(records)

    def resolve(self, chunk_id, version):
        if version != self.version or chunk_id not in self.documents:
            raise ValueError("stale/unknown corpus citation")
        return self.documents[chunk_id]

    def keyword(self, query, version, top_k=5):
        """Local lexical rehearsal only; never labelled hosted/vector retrieval."""
        if version != self.version or not 1 <= top_k <= 5 or len(query) > 1000:
            raise ValueError("invalid query/version/top-k")
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked = []
        for document in self.documents.values():
            tokens = set(
                re.findall(r"[a-z0-9]+", (document["title"] + " " + document["content"]).lower())
            )
            score = len(words & tokens)
            if score:
                ranked.append({**document, "score": score, "corpus_version": version})
        return sorted(ranked, key=lambda x: (-x["score"], x["id"]))[:top_k]


class SearchRunbooks:
    def __init__(
        self,
        search_client,
        embedding_client,
        deployment,
        dimensions,
        corpus,
        *,
        guard,
        deployment_version,
    ):
        self.search = search_client
        self.embeddings = embedding_client
        self.deployment = deployment
        self.dimensions = dimensions
        self.corpus = corpus
        self.guard = guard
        if not deployment_version:
            raise ValueError("pinned embedding deployment version required")
        self.deployment_version = deployment_version
        self.traces = []

    def embed(self, texts, *, timeout_seconds=30):
        if timeout_seconds <= 0:
            raise TimeoutError("embedding deadline exhausted")
        self.guard("embedding")
        if hasattr(self.guard, "embedding_inputs"):
            self.guard.embedding_inputs(texts)
        response = self.embeddings.embeddings.create(
            model=self.deployment,
            input=texts,
            dimensions=self.dimensions,
            timeout=timeout_seconds,
        )
        vectors = [row.embedding for row in sorted(response.data, key=lambda x: x.index)]
        array = np.asarray(vectors)
        if array.shape != (len(texts), self.dimensions) or not np.isfinite(array).all():
            raise ValueError("embedding shape/nonfinite output mismatch")
        self.traces.append(
            {
                "operation": "embedding",
                "deployment": self.deployment,
                "model": response.model,
                "tokens": response.usage.total_tokens,
                "input_sha256": identity(texts),
            }
        )
        return vectors

    def publish(self, cache_path: Path):
        self.guard("index_write")
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        documents = []
        for document in self.corpus.documents.values():
            key = identity(
                [self.deployment, self.deployment_version, self.dimensions, document["sha256"]]
            )
            vector = cache.get(key)
            if vector is None:
                vector = self.embed([document["content"]])[0]
                cache[key] = vector
            if len(vector) != self.dimensions or not np.isfinite(vector).all():
                raise ValueError("invalid cached embedding")
            documents.append({**document, "vector": vector, "corpus_version": self.corpus.version})
        result = self.search.upload_documents(documents=documents)
        if len(result) != len(documents) or any(not x.succeeded for x in result):
            raise ValueError("index publication partially failed")
        from fleet_fault_triage.data import save_json

        save_json(cache_path, cache)
        for document in documents:
            restored = self.search.get_document(key=document["id"])
            if (
                restored["sha256"] != document["sha256"]
                or restored["corpus_version"] != self.corpus.version
            ):
                raise ValueError("index document readback mismatch")
        return {
            "corpus_version": self.corpus.version,
            "chunks": len(documents),
            "traces": self.traces,
        }

    def query(self, query, version, top_k=5, mode="hybrid", *, timeout_seconds=30):
        if timeout_seconds <= 0:
            raise TimeoutError("retrieval deadline exhausted")
        deadline = time.monotonic() + timeout_seconds
        if version != self.corpus.version or not 1 <= top_k <= 5 or len(query) > 1000:
            raise ValueError("invalid query/version/top-k")
        if mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        self.guard("retrieval")
        options = {
            "top": top_k,
            "filter": f"corpus_version eq '{version}'",
            "select": ["id", "title", "content", "check", "sha256", "corpus_version"],
        }
        if mode != "keyword":
            from azure.search.documents.models import VectorizedQuery

            options["vector_queries"] = [
                VectorizedQuery(
                    vector=self.embed([query], timeout_seconds=timeout_seconds)[0],
                    k_nearest_neighbors=top_k,
                    fields="vector",
                )
            ]
        request_ids = []

        def response_hook(response):
            request_ids.append(response.http_response.headers.get("request-id"))

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("retrieval deadline exhausted after embedding")
        rows = list(
            self.search.search(
                search_text=query if mode != "vector" else None,
                raw_response_hook=response_hook,
                connection_timeout=min(10, remaining),
                read_timeout=remaining,
                **options,
            )
        )
        for row in rows:
            expected = self.corpus.resolve(row["id"], version)
            if row["sha256"] != expected["sha256"] or row["content"] != expected["content"]:
                raise ValueError("retrieved content differs from the approved corpus")
        self.traces.append(
            {
                "operation": "query",
                "mode": mode,
                "query_sha256": identity(query),
                "corpus_version": version,
                "returned_ids": [r["id"] for r in rows],
                "scores": [r.get("@search.score") for r in rows],
                "provider_request_ids": request_ids,
            }
        )
        return rows


def index_definition(name, dimensions):
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    return SearchIndex(
        name=name,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="check", type=SearchFieldDataType.String),
            SimpleField(name="sha256", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="corpus_version", type=SearchFieldDataType.String, filterable=True),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimensions,
                vector_search_profile_name="runbooks",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="runbooks", algorithm_configuration_name="hnsw")],
        ),
    )
