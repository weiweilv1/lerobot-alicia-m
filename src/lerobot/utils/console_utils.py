#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any


@contextmanager
def safe_console_output() -> Iterator[None]:
    """Temporarily escape characters unsupported by the active console encoding.

    Some hardware SDKs print Unicode symbols during construction. On Windows consoles using a legacy
    code page such as GBK, those prints can otherwise raise ``UnicodeEncodeError`` before the device is
    even connected. Streams without ``reconfigure`` support are left untouched.
    """
    changed_streams: list[tuple[Any, str | None]] = []
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        old_errors = getattr(stream, "errors", None)
        try:
            reconfigure(errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            continue
        changed_streams.append((stream, old_errors))

    try:
        yield
    finally:
        for stream, old_errors in changed_streams:
            if old_errors is None:
                continue
            with suppress(AttributeError, OSError, ValueError):
                stream.reconfigure(errors=old_errors)
