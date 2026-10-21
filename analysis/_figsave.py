#!/usr/bin/env python3
"""Deterministic figure saving.

matplotlib stamps a `CreationDate` into every PDF it writes, so two runs of the
same script over the same data produce PDFs that differ in exactly those bytes.
That one field would otherwise make `MANIFEST.sha256` unverifiable after a run.

`savefig()` here pins the date instead of letting it default to "now":

  * if `SOURCE_DATE_EPOCH` is set, that UTC instant is used (the reproducible-builds
    convention, the same variable other reproducible builds use);
  * otherwise a fixed epoch, `FIXED_EPOCH`, is used.

Either way the PDF is a pure function of the data. PNGs
carry no timestamp and need none of this, but they go through the same call so
that every figure is written one way.
"""
from __future__ import annotations
import datetime as _dt, os

# 2026-01-01T00:00:00Z. Changing it changes every PDF's bytes.
FIXED_EPOCH = 1767225600


def creation_date() -> _dt.datetime:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", FIXED_EPOCH))
    return _dt.datetime.fromtimestamp(epoch, _dt.timezone.utc)


def savefig(fig, path, omit_date=False, **kw):
    """savefig with a pinned CreationDate for PDFs. Returns `path`.

    `omit_date=True` leaves the field out altogether, which is what the committed
    PDFs carry.
    """
    path = str(path)
    if path.endswith(".pdf"):
        md = dict(kw.pop("metadata", None) or {})
        md.setdefault("CreationDate", None if omit_date else creation_date())
        fig.savefig(path, metadata=md, **kw)
    else:
        fig.savefig(path, **kw)
    return path


if __name__ == "__main__":
    print(f"FIXED_EPOCH = {FIXED_EPOCH}  ->  {creation_date().isoformat()}")
    print("SOURCE_DATE_EPOCH =", os.environ.get("SOURCE_DATE_EPOCH", "(unset)"))
