"""bearer token 鉴权依赖（常量时间比对）。"""

from __future__ import annotations

import secrets
from typing import Callable

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


def make_token_guard(expected: str) -> Callable[[HTTPAuthorizationCredentials], None]:
    """生成一个校验函数：token 不匹配抛 401。常量时间比对防时序攻击。

    Args:
        expected: 预期的 bearer token 字符串；为空字符串时永远拒绝（未配置即拒绝）。

    Returns:
        一个可直接用作 FastAPI Depends 的可调用对象；接受
        ``HTTPAuthorizationCredentials`` 参数，匹配时静默返回，不匹配时
        抛出 ``HTTPException(status_code=401)``。

    Example:
        >>> guard = make_token_guard("my-secret")
        >>> app = FastAPI()
        >>> @app.get("/data")
        ... def get_data(cred: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        ...     guard(cred)
        ...     return {"ok": True}
    """

    def guard(cred: HTTPAuthorizationCredentials) -> None:
        """校验 bearer token，不匹配时抛 401。

        Args:
            cred: FastAPI HTTPBearer 解析出的凭据对象。

        Raises:
            HTTPException: token 为空或与预期不符时，status_code=401。
        """
        if not expected or not secrets.compare_digest(cred.credentials, expected):
            raise HTTPException(status_code=401, detail="invalid bridge token")

    return guard
