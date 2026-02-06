"""
Notion API 연동 모듈.

이 모듈은 Notion 워크스페이스의 페이지와 데이터베이스를 RAG 시스템에
마이그레이션하기 위한 비동기 클라이언트를 제공합니다.

┌──────────────────────────────────────────────────────────────────────┐
│                    Notion API 데이터 구조 개요                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Notion 워크스페이스                                                  │
│  └── 페이지 (Page)                                                   │
│       ├── 속성 (Properties): 제목, 생성일, 수정일 등 메타데이터       │
│       └── 블록 (Blocks): 실제 콘텐츠를 담는 계층 구조                 │
│            ├── paragraph: 일반 텍스트                                │
│            ├── heading_1/2/3: 제목                                   │
│            ├── bulleted_list_item: 불릿 목록                         │
│            ├── numbered_list_item: 번호 목록                         │
│            ├── to_do: 체크박스 항목                                  │
│            ├── code: 코드 블록                                       │
│            ├── quote: 인용문                                         │
│            ├── callout: 강조 박스                                    │
│            ├── divider: 구분선                                       │
│            └── child_page: 하위 페이지 (재귀 탐색 필요)              │
│                                                                      │
│  각 블록은 has_children 플래그로 하위 블록 존재 여부 표시             │
│  → True인 경우 별도 API 호출로 children 조회 필요                     │
│                                                                      │
│  텍스트는 rich_text 배열로 제공 (서식 정보 포함)                      │
│  → plain_text 필드에서 순수 텍스트만 추출                             │
└──────────────────────────────────────────────────────────────────────┘

핵심 기능:
- 페이지 콘텐츠 조회 및 재귀적 블록 탐색
- 블록 타입별 마크다운 변환
- Document 객체로 변환하여 벡터 스토어에 저장 가능
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import aiohttp
from common.config import settings
from models.state import Document

logger = logging.getLogger(__name__)


class NotionConnector:
    """
    비동기 Notion API 연동 클라이언트.

    이 클래스는 Notion API와 통신하여 페이지 콘텐츠를 가져오고,
    RAG 시스템에서 사용할 수 있는 Document 형식으로 변환합니다.

    설계 원칙:
    - 세션/헤더 속성명을 _session/_headers로 단일화
    - 모든 I/O 요청에 timeout 명시 (기본 15초)
    - print 대신 구조화 로그(extra) 사용 (JSON 로깅 호환)
    - 세션은 지연 생성(lazy)하며 close()로 명시적 종료

    Attributes:
        BASE_URL: Notion API 기본 URL
        _session: aiohttp 클라이언트 세션 (지연 생성)
        _timeout: API 요청 타임아웃 설정
        _headers: 인증 토큰과 API 버전을 포함한 요청 헤더
    """

    BASE_URL = "https://api.notion.com/v1"

    def __init__(
        self,
        notion_token: str,
        notion_version: str,
        timeout_seconds: float = 15.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._session: aiohttp.ClientSession | None = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {notion_token}",
            "Notion-Version": notion_version,
            "Content-Type": "application/json",
        }

    async def __aenter__(self) -> NotionConnector:
        """
        비동기 컨텍스트 매니저 진입 - HTTP 세션 초기화.

        ┌──────────────────────────────────────────────────────────────┐
        │               리소스 생명주기 추적 (Telemetry)               │
        │ ──────────────────────────────────────────────────────────── │
        │ 진입 시점: async with NotionConnector(...) as connector:    │
        │ 할당 리소스:                                                 │
        │   - aiohttp.ClientSession (TCP 연결 풀)                     │
        │   - 인증 헤더 (Bearer 토큰)                                  │
        │   - 타임아웃 설정 (기본 15초)                                │
        │                                                             │
        │ 참고: 세션은 지연 생성(lazy)되며, 실제 요청 시 연결 수립    │
        └──────────────────────────────────────────────────────────────┘
        """
        logger.debug(
            "notion_connector_enter",
            extra={"connector_id": id(self), "timeout": self._timeout.total},
        )
        print(f"[NotionConnector] __aenter__: HTTP 세션 초기화 (id={id(self)})")
        await self._get_session()
        print(
            f"[NotionConnector] __aenter__: 세션 준비 완료 "
            f"(타임아웃: {self._timeout.total}초)"
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """
        비동기 컨텍스트 매니저 종료 - HTTP 세션 종료.

        ┌──────────────────────────────────────────────────────────────┐
        │               리소스 해제 보장 (메모리 누수 방지)            │
        │ ──────────────────────────────────────────────────────────── │
        │ 종료 시점: with 블록 종료 또는 예외 발생 시                  │
        │ 해제 리소스:                                                 │
        │   - aiohttp.ClientSession.close() 호출                      │
        │   - TCP 연결 풀 반환                                         │
        │   - 미완료 요청 취소                                         │
        │                                                             │
        │ 중요: close() 없이 프로세스 종료 시 "Unclosed session"      │
        │       경고 발생 및 잠재적 리소스 누수                        │
        └──────────────────────────────────────────────────────────────┘
        """
        logger.debug(
            "notion_connector_exit",
            extra={
                "connector_id": id(self),
                "exception_type": exc_type.__name__ if exc_type else None,
            },
        )
        print(f"[NotionConnector] __aexit__: 세션 종료 시작 (id={id(self)})")
        if exc_type:
            print(
                f"[NotionConnector] __aexit__: 예외 감지 - {exc_type.__name__}: {exc}"
            )
        await self.close()
        print("[NotionConnector] __aexit__: 세션 종료 완료 (TCP 연결 반환)")

    async def _get_session(self) -> aiohttp.ClientSession:
        """aiohttp 세션을 지연 생성하여 반환합니다."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        """세션 종료"""
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get_json(self, url: str) -> dict[str, Any] | None:
        """
        GET 요청 후 JSON 응답을 반환합니다.

        ┌──────────────────────────────────────────────────────────────────────┐
        │                     에러 핸들링 전략 상세                             │
        ├──────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Notion API 에러 유형 및 대응:                                        │
        │                                                                      │
        │  [HTTP 상태 코드 기반 에러]                                           │
        │  ┌─────────┬──────────────────────────────────────────────────────┐  │
        │  │ 400     │ 잘못된 요청 (파라미터 오류)                          │  │
        │  │ 401     │ 인증 실패 (토큰 만료/무효)                           │  │
        │  │ 403     │ 권한 없음 (페이지 접근 권한 부족)                    │  │
        │  │ 404     │ 리소스 없음 (삭제된 페이지/잘못된 ID)                │  │
        │  │ 429     │ 요청 한도 초과 (Rate Limit)                          │  │
        │  │ 500     │ Notion 서버 오류                                     │  │
        │  └─────────┴──────────────────────────────────────────────────────┘  │
        │                                                                      │
        │  → 모든 비-2xx 응답: 로그 기록 후 None 반환 (호출자가 처리)          │
        │  → 에러 텍스트는 1000자로 제한 (로그 크기 관리)                      │
        │                                                                      │
        │  [네트워크/클라이언트 에러]                                           │
        │  ┌─────────────────────────────────────────────────────────────────┐ │
        │  │ ClientConnectionError: 연결 실패 (DNS, 네트워크 단절)          │ │
        │  │ ClientResponseError: 응답 처리 실패                            │ │
        │  │ ClientPayloadError: 응답 본문 파싱 실패                        │ │
        │  │ TimeoutError: 요청 타임아웃 (기본 15초)                        │ │
        │  └─────────────────────────────────────────────────────────────────┘ │
        │                                                                      │
        │  → aiohttp.ClientError로 일괄 캐치                                   │
        │  → 스택 트레이스 포함 로그 (logger.exception)                        │
        │  → None 반환으로 상위 로직이 graceful하게 처리                       │
        │                                                                      │
        │  ※ 재시도(retry) 로직은 의도적으로 미포함                            │
        │    → 호출자가 필요 시 tenacity 등으로 래핑                           │
        └──────────────────────────────────────────────────────────────────────┘

        Args:
            url: 요청할 Notion API URL

        Returns:
            성공 시 JSON 딕셔너리, 실패 시 None
        """
        session = await self._get_session()

        try:
            async with session.get(url, headers=self._headers) as response:
                # ═══════════════════════════════════════════════════════
                # HTTP 상태 코드 검증: 2xx가 아니면 실패로 처리
                # ═══════════════════════════════════════════════════════
                if response.status < 200 or response.status >= 300:
                    error_text = await response.text()
                    logger.warning(
                        "notion_api_non_2xx",
                        extra={
                            "url": url,
                            "status": response.status,
                            "error_text": error_text[:1000],  # 로그 크기 제한
                        },
                    )
                    return None
                return await response.json()

        except aiohttp.ClientError as exc:
            # ═══════════════════════════════════════════════════════════
            # 네트워크/클라이언트 레벨 에러 처리
            # - 연결 실패, 타임아웃, 응답 파싱 실패 등
            # - exception()으로 스택 트레이스 포함 로깅
            # ═══════════════════════════════════════════════════════════
            logger.exception(
                "notion_api_client_error",
                extra={"url": url, "error": str(exc)},
            )
            return None

    async def get_page_content(self, page_id: str) -> dict[str, Any] | None:
        """페이지 ID 기반으로 페이지 속성과 루트 블록(children)을 가져옵니다."""
        page_url = f"{self.BASE_URL}/pages/{page_id}"
        page_data = await self._get_json(page_url)
        if not page_data:
            return None

        blocks_url = f"{self.BASE_URL}/blocks/{page_id}/children"
        blocks_data = await self._get_json(blocks_url)
        if blocks_data and isinstance(blocks_data.get("results"), list):
            page_data["content"] = blocks_data["results"]
        else:
            page_data["content"] = []

        return page_data

    async def fetch_all_blocks(self, block_id: str) -> list[dict[str, Any]]:
        """
        블록의 모든 하위 블록을 재귀적으로 가져옵니다.

        ┌──────────────────────────────────────────────────────────────────────┐
        │                    재귀적 블록 탐색 로직                              │
        ├──────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Notion 블록 구조는 트리 형태:                                        │
        │                                                                      │
        │  페이지 (root)                                                       │
        │  ├── heading_1 (has_children: false)                                │
        │  ├── paragraph (has_children: false)                                │
        │  ├── toggle (has_children: true) ← 하위 블록 존재!                  │
        │  │   ├── paragraph                                                  │
        │  │   └── bulleted_list_item                                         │
        │  └── child_page (has_children: true) ← 하위 페이지!                 │
        │      └── (별도 페이지로 재귀 탐색)                                   │
        │                                                                      │
        │  알고리즘:                                                            │
        │  1. 현재 블록의 children API 호출                                    │
        │  2. 각 자식 블록을 순회                                              │
        │  3. has_children=true인 블록 발견 시 재귀 호출                       │
        │  4. 재귀 결과를 부모 블록의 "children" 키에 저장                     │
        │                                                                      │
        │  주의사항:                                                            │
        │  - 깊은 중첩 구조의 경우 API 호출이 많아질 수 있음                   │
        │  - Notion API Rate Limit (3 req/sec) 고려 필요                       │
        └──────────────────────────────────────────────────────────────────────┘

        Args:
            block_id: 탐색 시작 블록 ID

        Returns:
            모든 하위 블록 리스트 (중첩 구조 포함)
        """
        blocks_url = f"{self.BASE_URL}/blocks/{block_id}/children"
        blocks_data = await self._get_json(blocks_url)
        if not blocks_data:
            return []

        blocks = blocks_data.get("results", [])
        if not isinstance(blocks, list):
            return []

        all_blocks: list[dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue

            all_blocks.append(block)

            # ═══════════════════════════════════════════════════════════
            # 재귀 탐색: has_children=true인 블록의 하위 블록 조회
            # - toggle, column, synced_block 등이 children을 가질 수 있음
            # - 재귀 결과는 부모 블록의 "children" 키에 저장
            # ═══════════════════════════════════════════════════════════
            if block.get("has_children") is True and isinstance(block.get("id"), str):
                children = await self.fetch_all_blocks(block["id"])
                if children:
                    block["children"] = children

        return all_blocks

    def extract_text_content(self, blocks: list[dict[str, Any]]) -> str:
        """Notion 블록 목록에서 텍스트만 추출합니다."""
        parts: list[str] = []
        self._append_blocks_text(parts, blocks, ordered_index_start=1)
        return "".join(parts)

    def _append_blocks_text(
        self,
        parts: list[str],
        blocks: Iterable[dict[str, Any]],
        ordered_index_start: int,
    ) -> None:
        """
        Notion 블록을 마크다운 텍스트로 변환하여 parts에 추가.

        ┌──────────────────────────────────────────────────────────────────────┐
        │                 블록 타입별 마크다운 변환 규칙                         │
        ├──────────────────────────────────────────────────────────────────────┤
        │                                                                      │
        │  Notion 블록 타입     →    마크다운 출력                              │
        │  ─────────────────────────────────────────────────────────────────── │
        │  paragraph           →    {텍스트}\n\n                               │
        │  heading_1           →    # {텍스트}\n\n                             │
        │  heading_2           →    ## {텍스트}\n\n                            │
        │  heading_3           →    ### {텍스트}\n\n                           │
        │  bulleted_list_item  →    • {텍스트}\n                               │
        │  numbered_list_item  →    1. {텍스트}\n (자동 넘버링)                │
        │  to_do               →    [x] 또는 [ ] {텍스트}\n                    │
        │  code                →    ```{언어}\n{코드}\n```\n\n                 │
        │  quote               →    > {텍스트}\n\n                             │
        │  callout             →    {이모지} {텍스트}\n\n                      │
        │  divider             →    \n---\n\n                                  │
        │  child_page          →    \n\n### {제목} ###\n\n                     │
        │                                                                      │
        │  ※ rich_text 배열에서 plain_text만 추출하여 사용                     │
        │  ※ 하위 블록(children)은 재귀 호출로 처리                            │
        └──────────────────────────────────────────────────────────────────────┘
        """
        ordered_index = ordered_index_start

        for block in blocks:
            block_type = block.get("type")

            # child_page: 하위 페이지 제목을 섹션 헤더로 삽입
            if block_type == "child_page":
                child_title = block.get("child_page", {}).get("title")
                if isinstance(child_title, str) and child_title:
                    parts.append(f"\n\n### {child_title} ###\n\n")

            if block_type == "paragraph":
                self._append_rich_text(
                    parts, block.get("paragraph", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "heading_1":
                parts.append("# ")
                self._append_rich_text(
                    parts, block.get("heading_1", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "heading_2":
                parts.append("## ")
                self._append_rich_text(
                    parts, block.get("heading_2", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "heading_3":
                parts.append("### ")
                self._append_rich_text(
                    parts, block.get("heading_3", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "bulleted_list_item":
                parts.append("• ")
                self._append_rich_text(
                    parts, block.get("bulleted_list_item", {}).get("rich_text", [])
                )
                parts.append("\n")

            elif block_type == "numbered_list_item":
                parts.append(f"{ordered_index}. ")
                self._append_rich_text(
                    parts, block.get("numbered_list_item", {}).get("rich_text", [])
                )
                parts.append("\n")
                ordered_index += 1

            elif block_type == "to_do":
                todo = block.get("to_do", {})
                checked = bool(todo.get("checked", False))
                parts.append(f"[{'x' if checked else ' '}] ")
                self._append_rich_text(parts, todo.get("rich_text", []))
                parts.append("\n")

            elif block_type == "code":
                code = block.get("code", {})
                language = code.get("language")
                parts.append(f"```{language if isinstance(language, str) else ''}\n")
                self._append_rich_text(parts, code.get("rich_text", []))
                parts.append("\n```\n\n")

            elif block_type == "quote":
                parts.append("> ")
                self._append_rich_text(
                    parts, block.get("quote", {}).get("rich_text", [])
                )
                parts.append("\n\n")

            elif block_type == "callout":
                callout = block.get("callout", {})
                icon = callout.get("icon", {})
                emoji = "+++"
                if (
                    isinstance(icon, dict)
                    and icon.get("type") == "emoji"
                    and isinstance(icon.get("emoji"), str)
                ):
                    emoji = icon["emoji"]
                parts.append(f"{emoji} ")
                self._append_rich_text(parts, callout.get("rich_text", []))
                parts.append("\n\n")

            elif block_type == "divider":
                parts.append("\n---\n\n")

            children = block.get("children")
            if isinstance(children, list) and children:
                self._append_blocks_text(parts, children, ordered_index_start=1)

    def _append_rich_text(self, parts: list[str], rich_text: Any) -> None:
        if not isinstance(rich_text, list):
            return
        for item in rich_text:
            if isinstance(item, dict):
                text = item.get("plain_text")
                if isinstance(text, str) and text:
                    parts.append(text)

    def _extract_page_title(self, page_data: dict[str, Any]) -> str:
        """페이지 제목 추출"""
        properties = page_data.get("properties", {})
        if not isinstance(properties, dict):
            return "Untitled"

        for key in ("title", "Title", "Name", "name"):
            title_prop = properties.get(key)
            if not isinstance(title_prop, dict):
                continue
            if title_prop.get("type") != "title":
                continue
            title_array = title_prop.get("title", [])
            if not isinstance(title_array, list) or not title_array:
                continue
            return "".join(
                t.get("plain_text", "") for t in title_array if isinstance(t, dict)
            )

        return "Untitled"

    async def fetch_page_as_document(
        self,
        page_id: str,
        recursive: bool = True,
        additional_metadata: dict[str, Any] | None = None,
    ) -> Document | None:
        """Notion 페이지를 Document 객체로 변환합니다."""
        page_data = await self.get_page_content(page_id)
        if not page_data:
            return None

        page_title = self._extract_page_title(page_data)

        blocks = page_data.get("content", [])
        if not isinstance(blocks, list):
            blocks = []

        if recursive:
            all_blocks: list[dict[str, Any]] = []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                all_blocks.append(block)
                if block.get("has_children") is True and isinstance(
                    block.get("id"), str
                ):
                    children = await self.fetch_all_blocks(block["id"])
                    if children:
                        block["children"] = children
            blocks = all_blocks

        text_content = self.extract_text_content(blocks)

        metadata: dict[str, Any] = {
            "source": "notion",
            "page_id": page_id,
            "page_title": page_title,
            "page_url": f"https://www.notion.so/{page_id.replace('-', '')}",
            "created_time": page_data.get("created_time", ""),
            "last_edited_time": page_data.get("last_edited_time", ""),
        }
        if additional_metadata:
            metadata.update(additional_metadata)

        return Document(
            content=text_content,
            metadata=metadata,
            doc_id=f"notion_{page_id.replace('-', '')}",
        )


def build_notion_connector() -> NotionConnector:
    """설정값 기반으로 NotionConnector를 생성합니다."""
    return NotionConnector(
        notion_token=settings.NOTION_TOKEN,
        notion_version=settings.NOTION_VERSION,
        timeout_seconds=15.0,
    )


# 싱글톤 인스턴스
notion_connector = build_notion_connector()
