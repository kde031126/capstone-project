// 세션 구성 및 학습

import { apiFetch } from './api';

// --- Types (백엔드 Schemas 대응) ---

export interface SessionCreate {
  user_id: number;
  step_id: number;
  words_count: number;
}

export interface SessionResponse {
  session_id: number;
  user_id: number;
  step_id: number;
  target_words: number[];
  words_count: number;
  is_completed: number;
  started_at: string;
  ended_at: string | null;
}

export interface SessionDetailResponse {
  session_info: {
    session_id: number;
    user_id: number;
    step_id: number;
    is_completed: number;
    started_at: string;
    ended_at: string | null;
  };
  report: any; // AI 리포트 데이터 (OpenAI 결과)
  stats: {
    total_words: number;
    correct_words: number;
    accuracy_rate: number;
  };
  records: {
    record_id: number;
    word_id: number;
    is_correct: boolean;
    child_text: string;
    child_phonemes: string;
    errors: any[];
  }[];
}

// --- API Functions ---

/**
 * 1. 세션 시작 (POST /sessions/)
 * 오늘 학습할 단어 리스트를 추천받고 새 세션을 생성합니다.
 */
export const startSession = (payload: SessionCreate) => {
  return apiFetch<SessionResponse>('/sessions/', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
};

/**
 * 2. 세션 종료 (PATCH /sessions/{session_id})
 * 학습이 완료되었을 때 세션을 종료 상태로 변경합니다.
 */
export const endSession = (sessionId: number) => {
  return apiFetch<SessionResponse>(`/sessions/${sessionId}`, {
    method: 'PATCH',
  });
};

/**
 * 3. 세션 상세 및 리포트 조회 (GET /sessions/{session_id})
 * 특정 세션의 결과, 통계, 그리고 AI가 생성한 부모용 리포트를 가져옵니다.
 */
export const getSessionDetail = (sessionId: number) => {
  return apiFetch<SessionDetailResponse>(`/sessions/${sessionId}`, {
    method: 'GET',
  });
};