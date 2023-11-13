from typing import Any, Dict, List, Optional, Union

Primitive = Union[str, float, int, bool]
DictValues = Union[Primitive, List[Primitive]]
StringKeyedNestedDict = Dict[str, Optional[Union[Any, 'StringKeyedNestedDict']]]
