import os, sys  # 콤마 뒤 공백 없음, 한 줄에 여러 모듈 (isort/black 위반)
from pydantic import BaseModel


class test_model(BaseModel):  # 클래스명 소문자 (명명 규칙 위반)
    name: str = "제나"  # 콜론/등호 앞뒤 공백 없음
    age: int = 20


def unformatted_function(a, b):  # 불필요한 공백과 인자 사이 공백 부족
    result = a + b  # 연산자 앞뒤 공백 없음
    return result


print(unformatted_function(10, 20))
