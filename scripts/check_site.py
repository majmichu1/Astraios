#!/usr/bin/env python3
"""Sanity-check the static site in site/ before it is published.

Usage: ``check_site.py site``

For every ``index.html``: exactly one ``<h1>``, a ``<title>``, a meta
description, an absolute canonical URL that matches the page's path, Open
Graph and Twitter tags, ``lang="en"``, a viewport, an icon, JSON-LD that
parses, alt text on every image, and internal links and assets that resolve
to files in the site. The sitemap must list every page and nothing else.
Exits non-zero with a list of problems.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

BASE = "https://majmichu1.github.io/Astraios"
REQUIRED_META = (
    "description", "twitter:card", "twitter:title", "twitter:description", "twitter:image",
)
REQUIRED_OG = ("og:title", "og:description", "og:url", "og:image", "og:type")


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.lang = None
        self.title = ""
        self.h1 = 0
        self.meta: dict[str, str] = {}
        self.links: list[str] = []
        self.assets: list[str] = []
        self.canonical = None
        self.icon = False
        self.jsonld: list[str] = []
        self.images_without_alt = 0
        self._in_title = False
        self._in_ld = False
        self._buf = ""
        self.viewport = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.lang = a.get("lang")
        elif tag == "title":
            self._in_title = True
        elif tag == "h1":
            self.h1 += 1
        elif tag == "meta":
            key = a.get("name") or a.get("property")
            if key:
                self.meta[key] = a.get("content", "")
            if a.get("name") == "viewport":
                self.viewport = True
        elif tag == "link":
            rel = (a.get("rel") or "").split()
            if "canonical" in rel:
                self.canonical = a.get("href")
            if "icon" in rel or "apple-touch-icon" in rel:
                self.icon = True
            if "stylesheet" in rel or "describedby" in rel:
                self.assets.append(a.get("href", ""))
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "img":
            self.assets.append(a.get("src", ""))
            if a.get("alt") is None:
                self.images_without_alt += 1
        elif tag == "script" and a.get("type") == "application/ld+json":
            self._in_ld = True
            self._buf = ""

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.jsonld.append(self._buf)

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._in_ld:
            self._buf += data


def check(site: Path) -> list[str]:
    problems: list[str] = []
    pages = sorted(p for p in site.rglob("index.html"))
    files = {p.resolve() for p in site.rglob("*") if p.is_file()}
    titles: dict[str, str] = {}
    descs: dict[str, str] = {}
    routes = []
    for html in pages:
        rel = html.parent.relative_to(site).as_posix()
        route = "/" if rel == "." else f"/{rel}/"
        routes.append(route)
        p = Page()
        p.feed(html.read_text())
        tag = f"{route}:"
        if p.lang != "en":
            problems.append(f"{tag} lang is {p.lang!r}")
        if not p.title.strip():
            problems.append(f"{tag} no <title>")
        elif p.title in titles:
            problems.append(f"{tag} duplicate title of {titles[p.title]}")
        titles[p.title] = route
        if p.h1 != 1:
            problems.append(f"{tag} {p.h1} <h1> elements")
        for m in REQUIRED_META + REQUIRED_OG:
            if not p.meta.get(m):
                problems.append(f"{tag} missing meta {m}")
        d = p.meta.get("description", "")
        if d and d in descs:
            problems.append(f"{tag} duplicate description of {descs[d]}")
        descs[d] = route
        if p.canonical != f"{BASE}{route}":
            problems.append(f"{tag} canonical is {p.canonical!r}, expected {BASE}{route}")
        if p.meta.get("og:url") != p.canonical:
            problems.append(f"{tag} og:url differs from canonical")
        if not p.viewport:
            problems.append(f"{tag} no viewport meta")
        if not p.icon:
            problems.append(f"{tag} no icon link")
        if p.images_without_alt:
            problems.append(f"{tag} {p.images_without_alt} image(s) without alt")
        if "noindex" in p.meta.get("robots", ""):
            problems.append(f"{tag} public page carries noindex")
        for ld in p.jsonld:
            try:
                data = json.loads(ld)
            except json.JSONDecodeError as exc:
                problems.append(f"{tag} invalid JSON-LD: {exc}")
                continue
            if data.get("@type") == "SoftwareApplication":
                required = (
                    "name", "operatingSystem", "softwareVersion", "downloadUrl",
                    "license", "isAccessibleForFree", "featureList",
                )
                for key in required:
                    if key not in data:
                        problems.append(f"{tag} SoftwareApplication lacks {key}")
                if "aggregateRating" in data or "review" in data:
                    problems.append(f"{tag} JSON-LD contains rating/review claims")
        for href in p.links + p.assets:
            if not href or href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = href.split("#")[0]
            if target.startswith("/"):
                # 404.html uses absolute /Astraios/ paths
                prefix = "/Astraios/"
                target = target[len(prefix):] if target.startswith(prefix) else target[1:]
                path = site / target
            else:
                path = html.parent / target
            if path.is_dir():
                path = path / "index.html"
            if path.resolve() not in files:
                problems.append(f"{tag} broken internal link {href!r}")
    sitemap = site / "sitemap.xml"
    if not sitemap.exists():
        problems.append("sitemap.xml missing")
    else:
        locs = re.findall(r"<loc>([^<]+)</loc>", sitemap.read_text())
        expected = {f"{BASE}{r}" for r in routes}
        if set(locs) != expected:
            diff = sorted(set(locs) ^ expected)
            problems.append(f"sitemap and pages differ: {diff}")
        for loc in locs:
            if urlparse(loc).scheme != "https":
                problems.append(f"sitemap loc not absolute https: {loc}")
    for name in ("robots.txt", "llms.txt", ".nojekyll", "404.html"):
        if not (site / name).exists():
            problems.append(f"{name} missing")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    problems = check(Path(argv[1]))
    for p in problems:
        print("ERROR:", p)
    print("site OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
