"""Sole-owner UC alias changes with complete release binding and persisted readback."""

import io
import json
import re

from fleet_fault_triage.contracts import Release, canonical


class RegistryOwner:
    def __init__(self, client, model_name, manifests):
        self.client, self.name = client, model_name
        self.manifests = manifests

    def current(self):
        registered = self.client.get_registered_model(self.name)
        previous = registered.tags.get("fft_champion_release")
        if previous is None:
            if "champion" in (registered.aliases or {}):
                raise ValueError("unbound provider alias; owner reconciliation required")
            return None
        version = self.client.get_model_version_by_alias(self.name, "champion")
        stored = version.tags.get("fft_release_" + previous)
        if stored != previous:
            raise ValueError("provider alias has no complete release binding")
        release = self.manifests.get(previous)
        if (
            release.sha256 != previous
            or release.model_uri != f"models:/{self.name}/{version.version}"
        ):
            raise ValueError("provider alias/release identity mismatch")
        return previous

    def promote(self, release, expected_previous):
        if release.origin != "azure" or not release.model_uri.startswith(f"models:/{self.name}/"):
            raise ValueError("wrong registry model or synthetic release")
        if self.current() != expected_previous:
            raise ValueError("provider champion changed; no alias write")
        version = release.model_uri.rsplit("/", 1)[1]
        candidate = self.client.get_model_version(self.name, version)
        if candidate.run_id != release.model_run_id:
            raise ValueError("candidate run differs from the frozen release")
        self.manifests.put(release)
        self.client.set_model_version_tag(
            self.name, version, "fft_release_" + release.sha256, release.sha256
        )
        # UC has no compare-and-set alias API. A single model-owner identity/process
        # serializes this operation; any interruption leaves a detectable mismatch.
        if self.current() != expected_previous:
            raise ValueError("provider champion changed during preparation")
        self.client.set_registered_model_alias(self.name, "champion", version)
        self.client.set_registered_model_tag(self.name, "fft_champion_release", release.sha256)
        if self.current() != release.sha256:
            raise ValueError("provider promotion readback failed; stop and reconcile")


class VolumeManifests:
    """Owner-only append policy on an approved UC release volume, using Files API."""

    def __init__(self, files_client, directory):
        if not re.fullmatch(r"/Volumes/[a-z][a-z0-9_]*/[a-z][a-z0-9_]*/releases", directory):
            raise ValueError("dedicated approved release volume required")
        self.files, self.directory = files_client, directory

    def get(self, release_hash):
        if not re.fullmatch(r"[0-9a-f]{64}", release_hash):
            raise ValueError("invalid release artifact hash")
        response = self.files.download(f"{self.directory}/{release_hash}.json")
        release = Release.model_validate(json.loads(response.contents.read()))
        if release.sha256 != release_hash:
            raise ValueError("provider release artifact hash mismatch")
        return release

    def put(self, release):
        from databricks.sdk.errors import NotFound

        try:
            saved = self.get(release.sha256)
        except NotFound:
            self.files.upload(
                f"{self.directory}/{release.sha256}.json",
                io.BytesIO(canonical(release.model_dump()).encode()),
                overwrite=False,
            )
            saved = self.get(release.sha256)
        if saved != release:
            raise ValueError("provider release artifact persisted readback failed")
