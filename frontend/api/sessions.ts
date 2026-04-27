import { apiFetch } from './api';

// --- Types ---

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

// 녹음 결과 응답 타입
export interface RecordResponse {
  record_id: number;
  session_id: number;
  word_id: number;
  is_correct: boolean;
  child_text: string | null;
  child_phonemes: string | null;
  errors: any[];
  created_at: string;
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
  report: {
    overall_feedback: string;
    improvement_points: string[];
    recommended_practice: string;
  };
  stats: {
    total_words: number;
    correct_words: number;
    accuracy_rate: number;
  };
  records: any[];
}

// --- API Functions ---

/**
 * 1. 세션 시작
 * 수정 포인트: '/sessions/' -> '/sessions/sessions/'
 */
export const startSession = (payload: SessionCreate) => {
  return apiFetch<SessionResponse>('/sessions/sessions/', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
};

/**
 * 2. 녹음 데이터 전송
 */
export const uploadRecord = (sessionId: number, wordId: number, audioUri: string) => {
  const formData = new FormData();
  
  const filename = audioUri.split('/').pop() || 'recording.m4a';
  const match = /\.(\w+)$/.exec(filename);
  const type = match ? `audio/${match[1]}` : `audio/m4a`;

  // @ts-ignore
  formData.append('file', {
    uri: audioUri,
    name: filename,
    type,
  });

  // 백엔드 문서에 따라 쿼리 파라미터가 아닌 FormData에 넣어야 할 수도 있습니다.
  formData.append('session_id', sessionId.toString());
  formData.append('word_id', wordId.toString());

  return apiFetch<RecordResponse>(`/records/`, {
    method: 'POST',
    body: formData,
  });
};

/**
 * 3. 세션 종료
 * 수정 포인트: `/sessions/${sessionId}` -> `/sessions/sessions/${sessionId}`
 */
export const endSession = (sessionId: number) => {
  return apiFetch<SessionResponse>(`/sessions/sessions/${sessionId}`, {
    method: 'PATCH',
  });
};

/**
 * 4. 세션 상세 조회
 * 수정 포인트: `/sessions/${sessionId}` -> `/sessions/sessions/${sessionId}`
 */
export const getSessionDetail = (sessionId: number) => {
  return apiFetch<SessionDetailResponse>(`/sessions/sessions/${sessionId}`, {
    method: 'GET',
  });
};