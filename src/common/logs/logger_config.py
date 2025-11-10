# src/common/logs/logger_config.py

import logging
import sys


def setup_logging(level=logging.INFO):
    """
    애플리케이션 루트 로거를 설정합니다.
    모든 모듈에서 logging.getLogger(__name__)를 호출하면 이 설정을 상속받습니다.
    """

    # -----------------------------------------------------------------
    # ⚠️ 중요: 이 함수는 api/main.py, jobs/batch/run_notion_etl.py 등
    # 애플리케이션 "시작점(Entrypoint)"에서 딱 한 번만 호출되어야 합니다.
    # -----------------------------------------------------------------

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)-8s] [%(name)-25s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout  # 모든 로그를 콘솔(stdout)로 보냅니다.
    )

    # (선택) 너무 시끄러운 서드파티 라이브러리 로그 레벨 조정
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("✅ 중앙 로깅 설정이 완료되었습니다. (Level: %s)", logging.getLevelName(level))