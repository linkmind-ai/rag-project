"""
loguru 중앙 설정 모듈.

프로젝트 전체 로그를 이 모듈 한 곳에서 관리합니다.
사용법:
    from common.logger import logger
    logger.info("메시지")
    logger.debug("session={}, docs={}", session_id, len(docs))
    logger.error("에러 발생: {}", e)

로그 레벨 기준:
    DEBUG   : 리소스 생명주기, 내부 처리 흐름 (운영 시 비활성)
    INFO    : 주요 비즈니스 이벤트 (초기화 완료, 쿼리 처리 완료 등)
    WARNING : 비정상이지만 복구 가능한 상황 (연결 재시도, 빈 결과 등)
    ERROR   : 처리 실패, 예외 발생 (즉시 확인 필요)
"""

import sys

from loguru import logger

# 기본 핸들러 제거 후 재설정
logger.remove()

# ── 콘솔 핸들러 ─────────────────────────────────────────────────────────────
logger.add(
    sys.stderr,
    level="DEBUG",
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
    backtrace=True,  # 예외 발생 시 전체 스택 트레이스 출력
    diagnose=True,  # 예외 발생 시 변수 값 출력 (디버깅용)
)

# ── 파일 핸들러 (로테이션) ───────────────────────────────────────────────────
logger.add(
    "logs/rag_{time:YYYY-MM-DD}.log",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}",
    rotation="00:00",  # 매일 자정에 새 파일
    retention="7 days",  # 7일치 보관
    compression="zip",  # 오래된 로그 압축
    encoding="utf-8",
    backtrace=True,
    diagnose=False,  # 파일에는 변수 값 미출력 (보안)
)

__all__ = ["logger"]
