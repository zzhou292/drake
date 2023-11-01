#!/usr/bin/env python3

"""Regenerates the manifest.bzl list of packages and checksums."""

import gzip
import sys
import urllib.request

# This is NVidia's package listing for their download site.
# When upgrading Ubuntu versions and supporting both versions as an
# intermediate step, please use the URL which references the later version of
# Ubuntu.
URL = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz"  # noqa

CUDA_VERSION = "11-8"  # Use `-` not `.` for filename matching.
# This is the CUDA packages and their versions that we want to use.
# From
#    https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-major-component-versions
# or for less-that-latest versions, browse through here instead:
#    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/
# Minimum NVIDIA Linux driver for this version: 470.82.01
PACKAGE_VERSION_MAP = {
    "cuda-cccl": "11.8.89",
    "cuda-cudart": "11.8.89",
    "cuda-cudart-dev": "11.8.89",
    "cuda-nvcc": "11.8.89",
    "cuda-sanitizer": "11.8.86",
    "libnpp": "11.8.0.86",
    "libnpp-dev": "11.8.0.86",
    "libnvjpeg": "11.9.0.86",
    "libnvjpeg-dev": "11.9.0.86",
    "libcusolver": "11.4.1.48",
    "libcusolver-dev": "11.4.1.48",
    "libcublas": "11.11.3.6",
    "libcublas-dev": "11.11.3.6",
    "libcusparse": "11.7.5.86",
    "libcusparse-dev": "11.7.5.86",
}


def expected_filename(package_name):
    """Returns the filename of a package based on its wanted version."""
    package_version = PACKAGE_VERSION_MAP[package_name]
    return f"{package_name}-{CUDA_VERSION}_{package_version}-1_amd64.deb"


def extract_manifest(packages_data):
    """Given unzipped Packages.gz contents, returns the MANIFEST dict to be
    written into manifest.bzl."""
    wanted_filenames = {
        expected_filename(package_name): package_name
        for package_name in PACKAGE_VERSION_MAP
    }
    result = {}
    for one_package in packages_data.split("\n\n"):
        lines = one_package.split("\n")
        if not one_package:
            continue
        # Extract "Filename: " and "SHA256: " for each entry.
        package_details = {}
        for key in ["Filename", "SHA256"]:
            prefix = key + ": "
            if key == "Filename":
                # Prune the leading "./" from filenames.
                prefix += "./"
            match = [x for x in lines if x.startswith(prefix)]
            assert len(match) == 1, one_package
            package_details[key] = match[0][len(prefix):]
        # Skip entries that to not match our desired version.
        filename = package_details["Filename"]
        if filename in wanted_filenames:
            package_name = wanted_filenames[filename]
            result[package_name] = package_details

    assert len(result) == len(
        PACKAGE_VERSION_MAP
    ), f"Missing packages: {set(PACKAGE_VERSION_MAP) - set(result)}"

    return result


def make_bzl(manifest):
    """Given the MANIFEST dict, return the manifest.bzl file contents."""
    result = [
        "# -*- mode: python -*-",
        "# vi: set ft=python :",
        "",
        "# GENERATED FILE " + "DO NOT EDIT",
        "# To update this, run ./tools/workspace/cuda/manifest_regenerate.py.",
        "",
        "CUDA_VERSION = \"{}\"".format(CUDA_VERSION.replace("-", ".")),
        "",
        "MANIFEST = {",
    ]
    for name in sorted(manifest.keys()):
        result.extend(
            [
                '    "{}": {{'.format(name),
                '        "Filename": "{}",'.format(manifest[name]["Filename"]),
                '        "SHA256": "{}",  # {}'.format(
                    manifest[name]["SHA256"], "noqa"
                ),
                '    },',
            ]
        )
    result.append("}")
    return "\n".join(result) + "\n"


def main():
    # Read NVidia's Packages.gz list of debs + checksums.
    gzipped_contents = urllib.request.urlopen(URL).read()
    listing = gzip.decompress(gzipped_contents).decode("utf-8")

    # Scrape the data that we want.
    manifest = extract_manifest(listing)

    # Format it into buildifer-formatted bzl file.
    bzl = make_bzl(manifest)

    # Overwrite Anzu's source manifest.
    with open("tools/workspace/cuda/manifest-22.04.bzl", "w") as f:
        f.write(bzl)

    return 0


if __name__ == "__main__":
    sys.exit(main())
