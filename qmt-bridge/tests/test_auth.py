"""bearer 鉴权依赖测试。"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from qmt_bridge.auth import make_token_guard


def _cred(tok):
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=tok)


def test_correct_token_passes():
    guard = make_token_guard("s3cret")
    guard(_cred("s3cret"))  # 不抛即通过


def test_wrong_token_401():
    guard = make_token_guard("s3cret")
    with pytest.raises(HTTPException) as ei:
        guard(_cred("nope"))
    assert ei.value.status_code == 401
