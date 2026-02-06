from fastapi import APIRouter, HTTPException, status
from stores.memory_store import memory_store

router = APIRouter(prefix="/session", tags=["session"])


@router.get("")
async def get_sessions():
    """메모리에 올라간 세션 목록 조회"""
    try:
        sessions = await memory_store.get_all_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 조회 오류 발생: {str(e)}",
        )


@router.get("/{session_id}/history")
async def get_session_history(session_id: str):
    """특정 세션 대화 이력 조회"""
    try:
        messages = await memory_store.get_recent_messages(session_id)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in messages
            ],
            "count": len(messages),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"대화 이력 조회 오류 발생: {str(e)}",
        )


@router.delete("/{session_id}")
async def clear_session(session_id: str):
    """특정 세션 대화 이력 삭제"""
    try:
        success = await memory_store.clear_history(session_id)
        if success:
            return {
                "success": True,
                "message": f"세션 {session_id}의 이력이 삭제되었습니다.",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="세션을 찾을 수 없습니다.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 삭제 오류 발생: {str(e)}",
        )
