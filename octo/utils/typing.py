from typing import Any, Mapping, Sequence, Union

import jax

# random number 생성을 위한 Key, 필요할 때 분할해서 사용
PRNGKey = jax.random.KeyArray
# 트리구조, map과 유사
PyTree = Union[jax.typing.ArrayLike, Mapping[str, "PyTree"]]
Config = Union[Any, Mapping[str, "Config"]]
Params = Mapping[str, PyTree]
Data = Mapping[str, PyTree]
Shape = Sequence[int]
Dtype = jax.typing.DTypeLike
